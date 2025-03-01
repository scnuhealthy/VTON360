#! /bin/bash

set -exu
exp_dir="output"
cam_path="demo_data/campath.json"

# 1. reconstruction with 3DGS.
ns-train splatfactox \
    --output-dir ${exp_dir} \
    --experiment-name test \
    --max-num-iterations 20000 \
    --pipeline.model.cull-alpha-thresh 0.005 \
    --pipeline.model.max-opacity-loss-scale 2.0 \
    --pipeline.model.background_color white \
    --viewer.quit-on-train-completion True \
    nerfstudio-data --data demo_data/0024_00208 \

# use the newest checkpoint
ckpt_dir=$(ls ${exp_dir}/test/splatfactox | sort -r | head -n 1)
echo using checkpoint ${ckpt_dir}

# 2. render a video with the given camera path.
# there is no detail about the format of the camera path in NeRF Studio's docs,
# but you could export camera path manually using ns-view and then 
# edit the exported JSON file (focus on the field named `camera_path` in the JSON file) for a better result.
ns-render camera-path \
    --camera-path-filename $cam_path \
    --load-config ${exp_dir}/test/splatfactox/${ckpt_dir}/config.yml \
    --output-path ${exp_dir}/video.mp4 \


# 3. render each frame in the video.
ns-render camera-path \
    --output-format images \
    --camera-path-filename $cam_path \
    --load-config ${exp_dir}/test/splatfactox/${ckpt_dir}/config.yml \
    --output-path ${exp_dir}/frames/ \
