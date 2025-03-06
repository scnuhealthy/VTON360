from PIL import Image
import os
import numpy as np
from PIL import Image, ImageOps
import json
import argparse

def read_annot_info(annot_path):
    annot_info = json.load(open(annot_path))
    bbox = annot_info['annots'][0]['bbox']
    width = annot_info['width']
    if width == 4096 or width == 2448:
        for i in range(4):
            bbox[i] = bbox[i] // 2
    elif width == 2048:
        pass
    else:
        print('wrong annot size',img_path)
        f.write('wrong annot size '+img_path+'\n')
    return bbox

def adjust_bbox_aspect_ratio(bbox, target_aspect_ratio=(2, 3)):
    """
    调整 bbox 以使宽高比符合目标宽高比 (2:3)
    
    :param bbox: 原始的 bbox，格式为 (x1, y1, x2, y2)
    :param target_aspect_ratio: 目标宽高比，默认为 2:3
    :return: 调整后的 bbox (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    current_aspect_ratio = width / height
    
    # 目标宽高比
    target_w, target_h = target_aspect_ratio

    new_width = height * (target_w / target_h)
    x1 = int(x1 - (new_width - width) / 2)  # 调整x1和x2
    x2 = int(x2 + (new_width - width) / 2)

    # # 如果当前宽高比小于目标宽高比，需要增加宽度
    # if current_aspect_ratio < target_w / target_h:
    #     new_width = height * (target_w / target_h)
    #     x1 = int(x1 - (new_width - width) / 2)  # 调整x1和x2
    #     x2 = int(x2 + (new_width - width) / 2)
    # # 如果当前宽高比大于目标宽高比，需要增加高度
    # elif current_aspect_ratio > target_w / target_h:
    #     new_height = width / (target_w / target_h)
    #     y1 = int(y1 - (new_height - height) / 2)  # 调整y1和y2
    #     y2 = int(y2 + (new_height - height) / 2)

    bbox = (x1, y1, x2, y2)
    return bbox


def post_process(img1,img2,parse,bbox,new_parse):
    parse_array = np.array(parse)
    parse_head = (parse_array == 1) + \
                    (parse_array == 2) + \
                    (parse_array == 3)+ \
                    (parse_array == 11) 
    parse_shoe = (parse_array == 9) + \
                (parse_array == 10)
    parse_head += parse_shoe
    # parse_head = (1 -parse_head).astype(bool)       
    head_mask = Image.fromarray(parse_head)
    head_mask = head_mask.convert('L')

    # Step 1: 填充mask变成原图
    bbox = adjust_bbox_aspect_ratio(bbox[:4])
    left = int(bbox[0])
    top = int(bbox[1])
    bbox_width = int(bbox[2] - bbox[0])
    bbox_height = int(bbox[3] - bbox[1])
    head_mask = head_mask.resize((bbox_width, bbox_height))

    mask_new = Image.new('L', img1.size, 0)
    mask_new.paste(head_mask, (left,top))

    # mask_new.save('head_mask.png')

    # parse new
    # new_parse_padded = Image.new('L', img1.size, 0)
    # new_parse_padded.paste(new_parse, (left,top))
    new_parse = new_parse.resize((bbox_width, bbox_height))
    padding_left = left
    padding_top = top
    padding_right = int(img1.size[0] - bbox[2])
    padding_bottom = int(img1.size[1] - bbox[3])
    new_parse_padded = ImageOps.expand(new_parse, (padding_left, padding_top, padding_right, padding_bottom), fill=0)
    # new_parse_padded.save('new_parse.png')


    # # 计算出img2放置的位置（将768*576图片放在1024*1024图片的中心）
    # left = (img1.width - img2.width) // 2
    # top = (img1.height - img2.height) // 2

    # 创建一个新的图像对象用于放置img2，保持尺寸与img1一致，并用黑色填充
    img2_paste = Image.new('RGB', img1.size, (0, 0, 0, 0))
    img2 = img2.resize((bbox_width, bbox_height))
    # 将img2粘贴到新图像的中心
    img2_paste.paste(img2, (left, top))

    # img2_paste.save('img2_new.png')

    # 使用调整后的mask将img2_paste覆盖到img1上，mask内的区域不会被覆盖
    result = Image.composite(img1, img2_paste, mask_new)

    # 保存结果图片
    return result, new_parse_padded


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='script')

    # 添加参数
    parser.add_argument('--image_root', type=str)
    parser.add_argument('--output_root', type=str)

    # 解析参数
    args = parser.parse_args()

    img2_root = args.image_root
    output_root = args.output_root
    # img2_root = 'output/image_output_tryon_1025_22000_test_multi_3_all2_mvg_back/'
    # img2_root = 'image_output_tryon_1015_60000_test_multi_3_all2_mvg'
    dataset_root = '/GPUFS/sysu_gbli2_1/hzj/mvhumannet/processed_mvhumannet'
    print(img2_root, output_root)
    # output_root = 'test_thuman_mvg_post'

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for cloth_id in os.listdir(img2_root):
        if not os.path.exists(os.path.join(output_root, cloth_id)):
            os.makedirs(os.path.join(output_root, cloth_id))
        img2_cloth_root = os.path.join(img2_root, cloth_id)
        output_cloth_root = os.path.join(output_root, cloth_id)

        for img2_name in os.listdir(img2_cloth_root):
            if 'cond' in img2_name or 'parse' in img2_name:
                continue
            info = img2_name.split('_')
            data_id = info[0]
            frame_id = info[3]
            camera_id = info[4]
            img_name = '%s_%s_img.jpg' % (camera_id, frame_id)

            img1_name = os.path.join(data_id, 'images_lr', frame_id, img_name)
            parse_name = os.path.join(data_id, 'parse2', frame_id, img_name).replace('jpg','png')
            annot_name = os.path.join(data_id, 'annots', frame_id, img_name).replace('jpg', 'json')
            new_parse_name = 'parse_' + img2_name.replace('jpg','png')

            img1_path = os.path.join(dataset_root, img1_name)
            img2_path = os.path.join(img2_cloth_root, img2_name)
            parse_path = os.path.join(dataset_root, parse_name)
            annot_path = os.path.join(dataset_root, annot_name)
            output_path = os.path.join(output_cloth_root, img2_name)
            new_parse_path = os.path.join(img2_cloth_root, new_parse_name)
            output_parse_path = os.path.join(output_cloth_root, new_parse_name)

            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            parse = Image.open(parse_path)
            new_parse = Image.open(new_parse_path)
            bbox = read_annot_info(annot_path)

            save_ori_path = os.path.join(output_cloth_root, 'ori_' + img2_name)
            img1.save(save_ori_path)

            # img1.save('img1.jpg')
            # img2.save('img2.jpg')
            # new_parse.save('parse.png')

            output_image, new_parse_padded = post_process(img1,img2,parse,bbox,new_parse)
            output_image.save(output_path)
            # output_image.save('img3.jpg')
            new_parse_padded.save(output_parse_path)
            # exit()


# for i in range(16):
#     i = str(i).zfill(3)
#     img2_root = '/data1/hezijian/render_data/val/img_edited1/0018_%s' % i
#     img1_root = '/data1/hezijian/render_data/val/img/0018_%s' % i
#     parse_root = '/data1/hezijian/render_data/val/parse2/0018_%s' % i
#     output_root = '/data1/hezijian/render_data/val/img_edited2/0018_%s' % i

#     if not os.path.exists(output_root):
#         os.makedirs(output_root)

#     for img_name in os.listdir(img2_root):
#         img1_path = os.path.join(img1_root, img_name)
#         img2_path = os.path.join(img2_root, img_name)
#         parse_path = os.path.join(parse_root, img_name.replace('jpg','png'))
#         output_path = os.path.join(output_root, img_name)

#         img1 = Image.open(img1_path)
#         img2 = Image.open(img2_path)
#         parse = Image.open(parse_path)

#         post_process(img1,img2,parse,output_path)

