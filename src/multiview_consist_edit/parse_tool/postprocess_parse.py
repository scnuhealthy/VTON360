import sys
# sys.path.append('./')
from PIL import Image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
import os
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import argparse

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='script')

    # 添加参数
    parser.add_argument('root', type=str)

    # 解析参数
    args = parser.parse_args()

    # root = '/GPUFS/sysu_gbli2_1/hzj/animate/output/image_output_tryon_1025_22000_test_multi_3_all2_mvg_back/'
    root = args.root
    parsing_model = Parsing(0)
    cloth_ids = os.listdir(root)

    for cloth_subroot in cloth_ids[:]:
        print(cloth_subroot)
        images = os.listdir(os.path.join(root, cloth_subroot))

        for image in images:
            if 'cond' in image or 'parse' in image:
                continue
            human_img_path = os.path.join(root, cloth_subroot, image)
            human_img = Image.open(human_img_path)
            model_parse, _ = parsing_model(human_img.resize((384,512)))
            model_parse = model_parse.resize((576,768))
            model_parse_path = os.path.join(root, cloth_subroot, 'parse_'+image.replace('jpg','png'))
            # print(model_parse_path)
            model_parse.save(model_parse_path)