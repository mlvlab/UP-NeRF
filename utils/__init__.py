import torch


def extract_model_state_dict(ckpt_path, model_name="model", prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    checkpoint_ = {}
    if "state_dict" in checkpoint:  # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint["state_dict"]
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name) + 1 :]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print("ignore", k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name="model", prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)
