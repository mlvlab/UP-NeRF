# UP-NeRF: Unconstrained Pose-Prior-Free Neural Radiance Fields
**[Project Page](https://mlvlab.github.io/upnerf/) |
[Paper](https://arxiv.org/abs/2311.03784)**

Injae Kim*,
Minhyuk Choi*,
Hyunwoo J. Kim†.


This repository is an official implementation of the NeurIPS 2023 paper UP-NeRF (Unconstrained Pose-Prior-Free Neural Radiance Fields) using [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning).

# 🏗️ Installation
```
git clone https://github.com/mlvlab/UP-NeRF.git
cd UP-NeRF
```
We recommend using [Anaconda](https://www.anaconda.com/download) to set up the environment.
```
conda create -n upnerf python=3.8 -y
conda activate upnerf

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
```

# 💻 Quick Start
## Data download
### 1. Manual download
Download the Phototourism data of the scene that you want to train from [here](https://www.cs.ubc.ca/~kmyi/imw2020/data.html) and train/val split file of the data from [here](https://nerf-w.github.io/).

And place them as shown below.

```
UP-NeRF
├── data
│   └── phototourism
│       ├── brandenburg_gate
│       │   ├── brandenburg_gate.tsv
│       │   └── dense
│       ├── taj_mahal
│       │   ├── taj_mahal.tsv
│       │   └── dense
│       ...
...
```
### 2. Script
Or you can simply use our automated download script.
```
# Example
sh scripts/download_phototourism.sh brandenburg_gate
```
Scenes provided are {brandenburg_gate, british_museum, lincoln_memorial_statue, pantheon_exterior, sacre_coeur, st_pauls_cathedral, taj_mahal, trevi_fountain}

### 3. Custom Dataset
To run with your own dataset, please check the format of metadata data/example/metadata.json and configuration file configs/custom.yaml (You can omit c2w fields in metadata.json if pose evaluation is not necessary. In addition, c2w matrices must be right up back format)
You must put images in dense/images (mandatory for compatability).


## Data Preprocessing
Before training you need to save DINO feature maps and DPT mono-depth maps.

Initialize the external submodule. (Last sanity check on Jan 2st, 2024)
```
git submodule update --init --recursive
```

Run the script for preprocessing. (example of Brandenburg Gate scene)

```diff
sh ./preprocess/preprocess_all.sh brandenburg_gate
```

:exclamation: Our script includes downloading checkpoint of DPT. If download fails due to some reasons, you can do it manually by downloading it from [here](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt), and move the weight to "./DPT/weights/".


Additionally, we highly recommend saving the cache data.

Run (example)

```diff
python prepare_phototourism.py --config configs/brandenburg_gate.yaml
```

### Caching Custom Dataset
Script is slightly different, so we separate the script file.
```diff
sh ./preprocess/preprocess_all_custom.sh data/example # you have to specify root directory of dataset
python prepare_phototourism.py --config configs/example.yaml
```
 


## Training
```diff
# If you saved the cache data.
python train.py --config configs/brandenburg_gate.yaml

# If you did not save the cache data.
python train.py --config configs/brandenburg_gate.yaml phototourism.use_cache False
```


You can change the yaml file to change the scene. Check the config files in ./configs
# :mag_right: Evaluation
## Test time optimization
Use [tto.py](tto.py) for test time optimization. It optimizes camera poses and appearance embeddings for test images.

Run (example)
```
python tto.py \
  --result_dir ./outputs/brandenburg_gate/UP-NeRF \
  --ckpt last \
  --optimize_num -1 \
  --wandb
```
## Print results
[eval.py](eval.py) prints results (PSNR, SSIM, LPIPS, rotation & translation errors).

Run (example)
```
python eval.py \
  --result_dir ./outputs/brandenburg_gate/UP-NeRF \
  --ckpt last
```

## Custom datasets
Currently tto.py and eval.py are only compatible with phototourism datasets.
If you also need them for custom datasets, create new issue.

# 📂 Weights
You can download pretrained weights from [here](https://drive.google.com/drive/folders/1L4xvuqI8umHOr7ViFMxQT7AxgEOpC9Jc?usp=sharing).
(brandenburg_gate, sacre_coeur, taj_mahal, trevi_fountain)


# Cite
```bibtex
@inproceedings{kim2023upnerf,
  title={UP-NeRF: Unconstrained Pose-Prior-Free Neural Radiance Fields},
  author={Kim, Injae and Choi, Minhyuk and Kim, Hyunwoo J},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

# Acknowledge
Our code is based on the implementation of NeRF in the Wild ([NeRF-W](https://github.com/kwea123/nerf_pl/tree/nerfw/)) and BARF ([BARF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF)).
