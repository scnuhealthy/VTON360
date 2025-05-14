from PIL import Image
import os
import numpy as np
import torch
import cv2

def crop(human_img_orig):
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
    return human_img

def apply_mask(image, mask):
    # 确保image和mask有相同的形状
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have the same width and height")
    
    # 创建一个白色的背景图像
    white_background = np.ones_like(image) * 255
    
    # 将mask以外的区域变为白色
    masked_image = np.where(mask[:, :, np.newaxis], image, white_background)
    
    return masked_image

def apply_mask_and_enlarge(image, mask, scale_factor=1.5):
    # 确保image和mask有相同的形状
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have the same width and height")

    # 获取mask区域的坐标
    y_indices, x_indices = np.where(mask)

    # 找到mask区域的边界框
    if y_indices.size == 0 or x_indices.size == 0:
        raise ValueError("Mask does not contain any True values")

    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # 裁剪出mask区域
    masked_region = image[y_min:y_max+1, x_min:x_max+1]
    masked_region_mask = mask[y_min:y_max+1, x_min:x_max+1]

    # 放大mask区域
    enlarged_width = int(masked_region.shape[1] * scale_factor)
    enlarged_height = int(masked_region.shape[0] * scale_factor)

    # 检查放大后的区域是否超出图像范围
    if (enlarged_width > image.shape[1]) or (enlarged_height > image.shape[0]):
        scale_factor = 1.5
        enlarged_width = int(masked_region.shape[1] * scale_factor)
        enlarged_height = int(masked_region.shape[0] * scale_factor)

    # 检查放大后的区域是否超出图像范围
    if (enlarged_width > image.shape[1]) or (enlarged_height > image.shape[0]):
        scale_factor = 1.0
        enlarged_width = int(masked_region.shape[1] * scale_factor)
        enlarged_height = int(masked_region.shape[0] * scale_factor)

    enlarged_region = cv2.resize(masked_region, (enlarged_width, enlarged_height), interpolation=cv2.INTER_LINEAR)
    enlarged_mask = cv2.resize(masked_region_mask.astype(np.uint8), (enlarged_width, enlarged_height), interpolation=cv2.INTER_NEAREST)

    # 创建一个白色的背景图像
    white_background = np.ones_like(image) * 255

    # 计算新放大区域的中心位置
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    new_x_start = max(center_x - enlarged_width // 2, 0)
    new_y_start = max(center_y - enlarged_height // 2, 0)
    new_x_end = min(new_x_start + enlarged_width, image.shape[1])
    new_y_end = min(new_y_start + enlarged_height, image.shape[0])

    # 将放大后的图像贴回原图，只保留mask为True的部分
    region_to_copy = enlarged_region[:new_y_end-new_y_start, :new_x_end-new_x_start]
    mask_to_copy = enlarged_mask[:new_y_end-new_y_start, :new_x_end-new_x_start].astype(bool)
    white_background[new_y_start:new_y_end, new_x_start:new_x_end][mask_to_copy] = region_to_copy[mask_to_copy]

    return white_background

root = '/data0/hezijian/datasets/save_render_data_yw/all' # set your path here
cloth_root = os.path.join('/data0/hezijian', 'cloth')   # set your path here

if not os.path.exists(cloth_root):
    os.makedirs(cloth_root)

sub_folder_list = os.listdir(root)
sub_folder_list = sorted(sub_folder_list)[:]
# sub_folder_list = ['200197','200198','200199']
ok_num = 0
wrong_num = 0
for sub_folder in sub_folder_list:
    sub_folder_path = os.path.join(root, sub_folder)


    try:
        parse_sub_folder_path = os.path.join(root, sub_folder, 'parse2')
        image_folder_path = os.path.join(root, sub_folder, 'images')

        front = '000.jpg' 
        back = '040.jpg' 

        front_img = os.path.join(image_folder_path, front)
        back_img = os.path.join(image_folder_path, back)
        front_parse = os.path.join(parse_sub_folder_path, front.replace('jpg','png'))
        back_parse = os.path.join(parse_sub_folder_path, back.replace('jpg','png'))

        # if not os.path.exists(front_img) or not os.path.exists(back_img):
        #     continue

        # if not os.path.exists(front_parse) or not os.path.exists(back_parse):
        #     continue

        front_img = Image.open(front_img)
        back_img = Image.open(back_img)
        front_parse = Image.open(front_parse)
        back_parse = Image.open(back_parse)

        front_img = np.array(crop(front_img).resize((384,512)))
        back_img = np.array(crop(back_img).resize((384,512)))
        front_parse = np.array(front_parse.resize((384,512)))
        back_parse = np.array(back_parse.resize((384,512)))
        
        mask_front = front_parse == 4
        mask_back = back_parse == 4
        front_cloth = apply_mask_and_enlarge(front_img, mask_front, scale_factor=2.0)
        back_cloth = apply_mask_and_enlarge(back_img, mask_back, scale_factor=2.0)
        # front_cloth = apply_mask(front_img, mask_front)
        # back_cloth = apply_mask(back_img, mask_back)

        front_cloth_name = os.path.join(cloth_root, '%s_front.jpg'%sub_folder)
        back_cloth_name = os.path.join(cloth_root, '%s_back.jpg'%sub_folder)
        front_cloth = Image.fromarray(front_cloth).save(front_cloth_name)
        back_cloth = Image.fromarray(back_cloth).save(back_cloth_name)
        print('ok',front_cloth_name)
        ok_num +=1

    except:
        print('wrong',parse_sub_folder_path)
        wrong_num +=1
print('ok',ok_num)
print('wrong',wrong_num)