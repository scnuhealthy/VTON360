from PIL import Image
import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import argparse
import json

def load_label_json(json_file):
    # 加载JSON文件
    with open(json_file) as f:
        data = json.load(f)

    # 创建一张新的图像，大小和原始图像相同，背景设为0（背景）
    mask_image = Image.new('L', (data['imageWidth'], data['imageHeight']), 0)

    for shape in data['shapes']:
        # 创建一个可以在上面绘制的mask
        mask = Image.new('L', (data['imageWidth'], data['imageHeight']), 0)
        draw = ImageDraw.Draw(mask)

        # 获取标注的点坐标
        points = [(int(point[0]), int(point[1])) for point in shape['points']]

        # 绘制多边形填充为1，边缘也是1，这代表被标注的区域
        draw.polygon(points, outline=1, fill=1)

        # 合并到最终的mask图像中，使用逻辑或运算确保重叠部分仍然为1
        # 注意：如果有重叠的标注区域，这种方法将会忽略重叠，将其也视作被标注区域
        final_mask_array = np.logical_or(np.array(mask_image), np.array(mask)) 

        # 更新最终的mask图像
        mask_image = Image.fromarray(final_mask_array.astype('bool'))
    return mask_image

def post_process(img1,img2,parse,new_parse,label_mask=None):
    parse_array = np.array(parse)
    parse_head = (parse_array == 1) + \
                    (parse_array == 2) + \
                    (parse_array == 3)+ \
                    (parse_array == 11) 
    parse_shoe = (parse_array == 9) + \
                (parse_array == 10)
    parse_head += parse_shoe
    if label_mask is not None:
        original_width, original_height = label_mask.size
        target_width = 768
        crop_amount = (original_width - target_width) // 2
        left = crop_amount
        upper = 0
        right = original_width - crop_amount
        lower = original_height
        procees_label_mask = label_mask.crop((left, upper, right, lower))
        procees_label_mask = procees_label_mask.resize((384,512))
        parse_head += procees_label_mask     
    head_mask = Image.fromarray(parse_head)
    head_mask = head_mask.convert('L')

    # Step 1: 填充mask到512x512
    mask = head_mask
    new_width = 512
    new_height = 512

    # 计算上下填充量
    padding_top = (new_height - mask.height) // 2
    padding_bottom = new_height - mask.height - padding_top

    # 计算左右填充量
    padding_left = (new_width - mask.width) // 2
    padding_right = new_width - mask.width - padding_left

    # 应用填充
    mask_padded = ImageOps.expand(mask, (padding_left, padding_top, padding_right, padding_bottom), fill=0)

    # Step 2: resize mask到1024x1024
    mask_resized = mask_padded.resize((1024, 1024))
    # mask_resized.save('head_mask.png')

    new_width = 1024
    new_height = 1024

    # 计算上下填充量
    padding_top = (new_height - new_parse.height) // 2
    padding_bottom = new_height - new_parse.height - padding_top

    # 计算左右填充量
    padding_left = (new_width - new_parse.width) // 2
    padding_right = new_width - new_parse.width - padding_left

    # 应用填充
    new_parse_padded = ImageOps.expand(new_parse, (padding_left, padding_top, padding_right, padding_bottom), fill=0)

    # Step 2: resize mask到1024x1024
    new_parse_padded = new_parse_padded.resize((1024, 1024))
    # new_parse_padded.save('new_parse.png')
    if label_mask is not None:
        a_array = np.array(label_mask)
        b_array = np.array(new_parse_padded)
        b_palette = new_parse_padded.getpalette()
        b_array[a_array == 1] = 2
        new_parse_padded = Image.fromarray(b_array)
        new_parse_padded.putpalette(b_palette)
    # new_parse_padded.save('new_parse2.png')

    # new_parse_padded.save('new_parse.png')

    # 计算出img2放置的位置（将768*576图片放在1024*1024图片的中心）
    left = (img1.width - img2.width) // 2
    top = (img1.height - img2.height) // 2

    # 创建一个新的图像对象用于放置img2，保持尺寸与img1一致，并用黑色填充
    img2_paste = Image.new('RGB', img1.size, (0, 0, 0, 0))

    # 将img2粘贴到新图像的中心
    img2_paste.paste(img2, (left, top))

    # 使用调整后的mask将img2_paste覆盖到img1上，mask内的区域不会被覆盖
    result = Image.composite(img1, img2_paste, mask_resized)

    # 保存结果图片
    return result, new_parse_padded


# 打开两张图片
# img1 = Image.open('/data1/hezijian/render_data/val/img/0018_000/2.jpg')
# img2 = Image.open('0018_000_2.jpg')
# parse = Image.open('/data1/hezijian/render_data/val/parse2/0018_000/2.png')


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
    dataset_root = '/GPUFS/sysu_gbli2_1/hzj/save_render_data_yw/all'
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
            # if '0228' not in img2_name and '0009' not in img2_name:
            #     continue

            if 'cond' in img2_name or 'parse' in img2_name:
                continue

            data_id = img2_name.split('_')[0]
            frame_id = img2_name.split('_')[2]
            

            img1_name = os.path.join(data_id, 'images', frame_id)
            parse_name = os.path.join(data_id, 'parse2', frame_id).replace('jpg','png')
            labelme_name = os.path.join(data_id, 'head_json', frame_id).replace('jpg','json')
            new_parse_name = 'parse_' + img2_name.replace('jpg','png')

            img1_path = os.path.join(dataset_root, img1_name)
            img2_path = os.path.join(img2_cloth_root, img2_name)
            parse_path = os.path.join(dataset_root, parse_name)
            labelme_path = os.path.join(dataset_root, labelme_name)
            output_path = os.path.join(output_cloth_root, img2_name)
            new_parse_path = os.path.join(img2_cloth_root, new_parse_name)
            output_parse_path = os.path.join(output_cloth_root, new_parse_name)

            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            parse = Image.open(parse_path)
            new_parse = Image.open(new_parse_path)
            if os.path.exists(labelme_path):
                label_mask = load_label_json(labelme_path)
            else:
                label_mask = None

            save_ori_path = os.path.join(output_cloth_root, 'ori_' + img2_name)
            
            # img1.save('img0.jpg')
            img1.save(save_ori_path)

            # img1.save('img1.jpg')
            # img2.save('img2.jpg')
            # parse.save('parse.png')

            output_image, new_parse_padded = post_process(img1,img2,parse,new_parse,label_mask)
            output_image.save(output_path)
            new_parse_padded.save(output_parse_path)



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

