<p align="center">
  
  <h1 align="center"><strong>🎥 [CVPR 2025] VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction</strong></h3>

<p align="center">
    <a href="https://github.com/scnuhealthy/" class="name-link" target="_blank">Zijian He<sup>1</sup> </a>,
    <a href="https://github.com/Thyme-git/" class="name-link" target="_blank">Yuwei Ning<sup>2</sup> </a>,
    <a href="https://scholar.google.com/citations?user=ojgWPpgAAAAJ&hl=zh-CN&oi=ao/" class="name-link" target="_blank">Yipeng Qin<sup>3</sup></a>,
    <a href="https://wanggrun.github.io/" class="name-link" target="_blank">Guangrun Wang<sup>1</sup></a>,
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=4pg3rtYAAAAJ" class="name-link" target="_blank">Sibei Yang<sup>4</sup></a>,
    <a href="https://scholar.google.com/citations?user=Nav8m8gAAAAJ&hl=zh-CN&oi=ao" class="name-link" target="_blank">Liang lin<sup>1,5</sup></a>,
    <a href="http://guanbinli.com/" class="name-link" target="_blank">Guanbin Li<sup>1,5*</sup></a>
    <br>
    * Corresponding authors <sup>1</sup>Sun Yat-sen University, <sup>2</sup>Huazhong University of Science and Technology,
    <br>
    <sup>3</sup>Cardiff University, <sup>4</sup>ShanghaiTech University, <sup>5</sup>Peng Cheng Laboratory
</p>

<div align="center">

<!-- [![Badge with Logo](https://img.shields.io/badge/arXiv-2403.08733-red?logo=arxiv)
](https://arxiv.org/abs/2403.08733) -->
[![Badge with Logo](https://img.shields.io/badge/Project-Page-blue?logo=homepage)](https://scnuhealthy.github.io/VTON360/)
</div>

## ✨ News
- [03.01.2024] We release our code and demo data for 3D lifting.

## ⚙️ Installation

Clone the repo and create a new conda env.
```bash
git clone https://github.com/scnuhealthy/VTON360.git
conda create -n vton360 python=? (TO BE CHECKED)
conda activate vton360
```


### 1. Diffusion Dependency

### 2. NeRF Studio

* You could have a look at [NeRF Studio Installation](https://docs.nerf.studio/quickstart/installation.html) for more detail.

1. Install NeRF Studio
```bash
python -m pip install --upgrade pip==24.2
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio==1.0.0
ns-install-cli # Optional, for tab completion.
```

2. Install gsplat
```bash
pip install gsplat==0.1.2.1
```

3. Install our customized [Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html). See [`src/splatfactox/README.md`](./splatfactox/README.md) for more details.
```bash
cd src
pip install -e .
```


## 🗄️ Data

### Use Our Preprocessed Data

### Render from Thuman2.0

## :arrow_forward: Get Started

### 1. Multi-view Consistant Try-On

#### A. Download the checkpoint and pre-trained models
1. Put the checkpoint into 'src/multiview_consist_edit/checkpoints'
We provide two checkpoints: 'thuman_tryon_mvattn_multi/checkpoints-30000' and 'mvhumannet_tryon_mvattn_multi/checkpoints-40000'

2. Download [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) and [sd-vae-ft-mse](https://huggingface.co/diffusers/sd-vae-ft-mse) from Huggingface

3. set the path in the 'src/multiview_consist_edit/config'

4. Download the parsing checkpoint from [here](https://huggingface.co/spaces/yisol/IDM-VTON/tree/main/ckpt/humanparsing)

#### B. Image editing
```bash
cd src/multiview_consist_edit
python infer_tryon_multi.py
```
The edited results are saved into 'output_root'.

Since the limitation of GPU memory, set 'output_front=False' and rerun to get the back views prediction
```bash
python infer_tryon_multi.py
```

#### C. Post-precross
```bash
cd parse_tool
python postprocess_parse.py 'output_root'
cd ../
python postprocess_thuman.py --image_root 'output_root' --output_root 'output_post_root' 
or python postprocess_mvhumannet.py --image_root 'output_root' --output_root 'output_post_root' 
```

#### D. Training
```bash
acclerate config
acclerate launch train_tryon_multi.py
```

### 2. 3D Lifting

#### A. Prepare Your Data as [NeRF Studio's Dataset Format](https://docs.nerf.studio/quickstart/data_conventions.html#dataset-format)

You need to prepare three components for 3D lifting with NeRF Studio.

* `images`: multi-view images containing the target person.
* `mask`: mask for the target human in multi-view images.
* `transforms.json`: NeRF Studio's dataset configuration.

We provide a demo dataset in `src/demo_data/splatfactox_demo_data`. 

You can simply replace the `images` and `mask` directories if you use our try-on result.

You can refer to [NeRF Studio's Dataset Format](https://docs.nerf.studio/quickstart/data_conventions.html#dataset-format) for more details if you want to use your own data.

#### B. Run Our Customized Splatfacto

> [Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) is an implementation of [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). You can refer to [here](./src/splatfactox/README.md) for our customized version of splatfacto.


```bash
cd src
bash scripts/splatfactox.sh
```

## Citation
If you find this code or find the paper useful for your research, please consider citing:
```

```