# Offial Implementation of Out VTON360

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

3. Install our customized [Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html). See [`src/splatfactox/README.md` for more details.](./splatfactox/README.md)
```bash
cd src
pip install -e .
```


## 🗄️ Data

### Use Our Preprocessed Data

### Customize Your Data

## :arrow_forward: Get Started

### 1. Multi-view Consistant Try-On

### 2. 3D Lifting

```bash
cd src
bash scripts/splatfactox.sh
```

## Citation
If you find this code or find the paper useful for your research, please consider citing:
```

```