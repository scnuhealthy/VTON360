# 3D Lifting Code for Our VTON360

## About the Code

This is an Extended Version of [NeRF Studio's Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html). We add support for scenes with no background.

We add an opacity loss to the background area and force the opacity in of the background to zero. But if used in early step of the 3DGS reconstruction, the opacity loss may force all Gaussians to be zero-opacity and result in training failure. So we linearly increasing the coef of the opacity loss from 0 to ``max-opacity-loss-scale``. More details can be found in ``splatfactox/splatfactox.py:286``.

## Usage

A simple demo is provided in ``scripts/splatfactox.sh``

## TODO

- [] release the code for view auto-selection with z-score nomalization.