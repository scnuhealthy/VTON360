import sys
sys.path.append('./')
from PIL import Image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from utils_mask import get_mask_location
import os
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import apply_net

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

def get_agnostic_mask(human_img_orig, is_checked_crop=True):
    openpose_model.preprocessor.body_estimation.model.to(device)

    human_img_orig = human_img_orig.convert("RGB")
    if is_checked_crop:
        human_img_orig = human_img_orig.resize((1024,1024))
        original_width, original_height = human_img_orig.size
        target_width = 768
        crop_amount = (original_width - target_width) // 2
        left = crop_amount
        upper = 0
        right = original_width - crop_amount
        lower = original_height
        cropped_image = human_img_orig.crop((left, upper, right, lower))
        cropped_width, cropped_height = cropped_image.size
        human_img = cropped_image
    else:
        human_img = human_img_orig.resize((768,1024))

    keypoints = openpose_model(human_img.resize((384,512)))
    model_parse, _ = parsing_model(human_img.resize((384,512)))

    mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
    mask = mask.resize((768,1024))

    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args,human_img_arg)
    pose_img = pose_img[:,:,::-1]
    pose_img = Image.fromarray(pose_img)

    return mask_gray, model_parse, pose_img

# human_img_orig = Image.open('gradio_demo/example/human/4_hr.jpg')
# # human_img_orig = Image.open('gradio_demo/example/human/00035_00.jpg')
# garm_img = Image.open('gradio_demo/example/cloth/09163_00.jpg')

# mask_gray, model_parse, pose_img = get_agnostic_mask(human_img_orig)
# mask_gray.save('1.jpg')

# model_parse = model_parse.convert('RGB')
# model_parse.save('2.jpg')

# pose_img.save('3.jpg')

root = '/PATH/TO/THUMAN2.1/all'

sub_folder_list = os.listdir(root)
sub_folder_list = sorted(sub_folder_list)
f = open('wrong_all_yw_512_600.txt','w')

for sub_folder in sub_folder_list[512:600]:
    sub_folder_path = os.path.join(root, sub_folder, 'images')
    parse_sub_folder_path = os.path.join(root, sub_folder, 'parse2')
    agnostic_sub_folder_path = os.path.join(root, sub_folder, 'agnostic')
    pose_sub_folder_path = os.path.join(root, sub_folder, 'pose')

    if not os.path.exists(parse_sub_folder_path):
        os.makedirs(parse_sub_folder_path)
    if not os.path.exists(agnostic_sub_folder_path):
        os.makedirs(agnostic_sub_folder_path)
    if not os.path.exists(pose_sub_folder_path):
        os.makedirs(pose_sub_folder_path)
    print(sub_folder)
    img_names = sorted(os.listdir(sub_folder_path))
    img_names = img_names[:80]
    for img_name in img_names:
        img_path = os.path.join(sub_folder_path, img_name)
        human_img_orig = Image.open(img_path)

        try:
            mask_gray, model_parse, pose_img = get_agnostic_mask(human_img_orig)
        
            mask_gray_path = os.path.join(agnostic_sub_folder_path, img_name)
            model_parse_path = os.path.join(parse_sub_folder_path, img_name.replace('jpg','png'))
            pose_img_path = os.path.join(pose_sub_folder_path, img_name)

            mask_gray.save(mask_gray_path)
            model_parse.save(model_parse_path)
            pose_img.save(pose_img_path)
            
        except:
            print('wrong',img_path)
            f.write(img_path+'\n')
            f.flush()
