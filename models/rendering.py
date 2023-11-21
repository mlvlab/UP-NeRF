import torch
from einops import rearrange, reduce, repeat

__all__ = ["render_rays"]


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, "n1 n2 -> n1 1", "sum")  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(
        torch.stack([below, above], -1), "n1 n2 c -> n1 (n2 c)", c=2
    )
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), "n1 (n2 c) -> n1 n2 c", c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), "n1 (n2 c) -> n1 n2 c", c=2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    denom[denom < eps] = 1
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )
    return samples


def render_rays(
    models,
    embeddings,
    rays,
    img_idx,
    sched_mult,
    N_samples=64,
    use_disp=False,
    perturb=0,
    N_importance=0,
    test_time=False,
    encode_feat=True,
    **kwargs,
):
    """
    Render rays by computing the output of @model applied on @rays and @ts
    Inputs:
        models: dict of NeRF models (coarse and fine) defined in nerf.py
        embeddings: dict of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3), ray origins and directions
        img_idx: (N_rays), ray time as embedding index
        sched_mult: schedule mult for candidate head
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, xyz, z_vals, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points on each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        """
        typ = model.typ
        N_samples_ = xyz.shape[1]

        inputs = {}
        inputs["input_xyz"] = rearrange(xyz, "n1 n2 c -> (n1 n2) c", c=3)
        inputs["input_dir"] = repeat(
            input_dir, "n1 c -> (n1 n2) c", n2=N_samples_
        ).detach()
        # create other necessary inputs
        if model.encode_appearance:
            inputs["input_a"] = repeat(a_embedded, "n1 c -> (n1 n2) c", n2=N_samples_)
        if model.encode_candidate:
            inputs["input_c"] = repeat(c_embedded, "n1 c -> (n1 n2) c", n2=N_samples_)

        outputs = model(inputs, sched_mult=sched_mult)
        for k, v in outputs.items():
            if "sigma" in k:
                outputs[k] = rearrange(
                    v, "(n1 n2) 1 -> n1 n2", n1=N_rays, n2=N_samples_
                )
            else:
                outputs[k] = rearrange(
                    v, "(n1 n2) c -> n1 n2 c", n1=N_rays, n2=N_samples_
                )

        # Convert these values using volume rendering
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e2 * torch.ones_like(
            deltas[:, :1]
        )  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
        s_alphas = 1 - torch.exp(-deltas * outputs["s_sigma"])  # shared(static) sigma
        if test_time and typ == "nerf_coarse":
            return
        if sched_mult < 1:
            if not model.encode_candidate:  # Do not use candidate head
                alphashifted = torch.cat(
                    [torch.ones_like(s_alphas[:, :1]), 1 - s_alphas], -1
                )  # [1, 1-a1, 1-a2, ...]
                transmittance = torch.cumprod(
                    alphashifted[:, :-1], -1
                )  # [1, 1-a1, (1-a1)(1-a2), ...]
                weights = s_alphas * transmittance
                if encode_feat:
                    results[f"s_weights_{typ}"] = weights
                    results[f"feat_{typ}"] = reduce(
                        rearrange(weights, "n1 n2 -> n1 n2 1") * outputs["s_feat"],
                        "n1 n2 c -> n1 c",
                        "sum",
                    )
                else:
                    raise NotImplemented
            else:
                c_alphas = 1 - torch.exp(-deltas * outputs["c_sigma"])
                alphas = 1 - torch.exp(
                    -deltas * (outputs["s_sigma"] + outputs["c_sigma"])
                )
                alphas_shifted = torch.cat(
                    [torch.ones_like(alphas[:, :1]), 1 - alphas], -1
                )  # [1, 1-a1, 1-a2, ...]
                transmittance = torch.cumprod(
                    alphas_shifted[:, :-1], -1
                )  # [1, 1-a1, (1-a1)(1-a2), ...]
                s_weights = s_alphas * transmittance
                c_weights = c_alphas * transmittance
                weights = alphas * transmittance
                results[f"c_weights_{typ}"] = weights
                results[f"c_depth_{typ}"] = reduce(
                    weights * z_vals, "n1 n2 -> n1", "sum"
                )
                if encode_feat:
                    s_feat_map = reduce(
                        rearrange(s_weights, "n1 n2 -> n1 n2 1") * outputs["s_feat"],
                        "n1 n2 c -> n1 c",
                        "sum",
                    )
                    c_feat_map = reduce(
                        rearrange(c_weights, "n1 n2 -> n1 n2 1") * outputs["c_feat"],
                        "n1 n2 c -> n1 c",
                        "sum",
                    )
                    results[f"feat_{typ}"] = s_feat_map + c_feat_map
                    results[f"t_weight_{typ}"] = reduce(c_weights, "n1 n2 -> n1", "sum")
                else:
                    s_rgb_map = reduce(
                        rearrange(s_weights, "n1 n2 -> n1 n2 1") * outputs["s_rgb"],
                        "n1 n2 c -> n1 c",
                        "sum",
                    )
                    c_rgb_map = reduce(
                        rearrange(c_weights, "n1 n2 -> n1 n2 1") * outputs["c_rgb"],
                        "n1 n2 c -> n1 c",
                        "sum",
                    )
                    results[f"c_rgb_{typ}"] = s_rgb_map + c_rgb_map
                    results[f"t_weight_{typ}"] = reduce(c_weights, "n1 n2 -> n1", "sum")
        if sched_mult > 0:
            s_alphas_shifted = torch.cat(
                [torch.ones_like(s_alphas[:, :1]), 1 - s_alphas], -1
            )
            s_transmittance = torch.cumprod(s_alphas_shifted[:, :-1], -1)
            only_s_weights = s_alphas * s_transmittance
            results[f"s_weights_{typ}"] = only_s_weights
            if test_time and typ == "nerf_coarse":
                return
            s_rgb_map = reduce(
                rearrange(only_s_weights, "n1 n2 -> n1 n2 1") * outputs["s_rgb"],
                "n1 n2 c -> n1 c",
                "sum",
            )
            results[f"s_rgb_{typ}"] = s_rgb_map

        s_alphas_shifted = torch.cat(
            [torch.ones_like(s_alphas[:, :1]), 1 - s_alphas], -1
        )
        s_transmittance = torch.cumprod(s_alphas_shifted[:, :-1], -1)
        only_s_weights = s_alphas * s_transmittance
        results[f"s_depth_{typ}"] = reduce(
            only_s_weights * z_vals, "n1 n2 -> n1", "sum"
        )
        return

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
    # Embed direction
    input_dir = rays_d

    rays_o = rearrange(rays_o, "n1 c -> n1 1 c")
    rays_d = rearrange(rays_d, "n1 c -> n1 1 c")

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")

    results = {}
    model = models["nerf_coarse"]
    if model.encode_candidate:
        c_embedded = embeddings["coarse_c"](img_idx)
    if model.encode_appearance:
        a_embedded = embeddings["coarse_a"](img_idx)
    inference(results, model, xyz_coarse, z_vals, test_time, **kwargs)

    # fine model
    if N_importance > 0:  # sample points for fine model
        model = models["nerf_fine"]
        z_vals_mid = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # (N_rays, N_samples-1) interval mid points
        if model.encode_candidate:
            if sched_mult == 0:
                z_vals_ = sample_pdf(
                    z_vals_mid,
                    results["c_weights_coarse"][:, 1:-1].detach(),
                    N_importance,
                    det=(perturb == 0),
                )
                z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
            elif sched_mult > 0 and sched_mult < 1:
                s_N_importance = round(sched_mult * N_importance)
                z_vals_ = sample_pdf(
                    z_vals_mid,
                    results["c_weights_coarse"][:, 1:-1].detach(),
                    N_importance - s_N_importance,
                    det=(perturb == 0),
                )
                s_z_vals_ = sample_pdf(
                    z_vals_mid,
                    results["s_weights_coarse"][:, 1:-1].detach(),
                    s_N_importance,
                    det=(perturb == 0),
                )
                z_vals = torch.sort(torch.cat([z_vals, s_z_vals_, z_vals_], -1), -1)[0]
            elif sched_mult == 1:
                z_vals_ = sample_pdf(
                    z_vals_mid,
                    results["s_weights_coarse"][:, 1:-1].detach(),
                    N_importance,
                    det=(perturb == 0),
                )
                z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
        else:
            z_vals_ = sample_pdf(
                z_vals_mid,
                results["s_weights_coarse"][:, 1:-1].detach(),
                N_importance,
                det=(perturb == 0),
            )
            # detach so that grad doesn't propogate to weights_coarse from here
            z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
        xyz_fine = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")
        if model.encode_candidate:
            c_embedded = embeddings["fine_c"](img_idx)
        if model.encode_appearance:
            a_embedded = embeddings["fine_a"](img_idx)
        inference(results, model, xyz_fine, z_vals, test_time, **kwargs)
    return results
