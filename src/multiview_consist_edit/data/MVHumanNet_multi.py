import os, io, csv, math, random
import numpy as np
from PIL import Image,ImageDraw
import json
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from transformers import CLIPProcessor
import random
from torchvision.transforms import functional as F
import torch.distributed as dist
import copy
import cv2
import pickle
from .camera_utils import read_camera_mvhumannet

def crop_and_resize(img, bbox, size):

    # 计算中心点和新的宽高
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    new_height = bbox[3] - bbox[1]
    new_width = int(new_height * (2 / 3))

    # 计算新的边界框
    new_bbox = [
        int(center_x - new_width / 2),
        int(center_y - new_height / 2),
        int(center_x + new_width / 2),
        int(center_y + new_height / 2)
    ]

    # 裁剪图像
    cropped_img = img.crop(new_bbox)

    # 调整大小
    resized_img = cropped_img.resize(size)

    return resized_img


class MVHumanNet_Dataset(Dataset):
    def __init__(
        self, dataroot, sample_size=(512,384), is_train=True, mode='pair', clip_model_path='', multi_length=8, output_front=True,
    ):
        im_names = []
        self.dataroot = os.path.join(dataroot, 'processed_mvhumannet')
        self.cloth_root = os.path.join(dataroot, 'cloth')
        self.data_ids = []
        self.data_frame_ids = []
        self.cloth_ids = []
        self.cloth_frame_ids = []
        if is_train:
            f = open(os.path.join(dataroot,'train_frame_ids.txt'))
            for line in f.readlines():
                line_info = line.strip().split()
                self.data_ids.append(line_info[0])
                self.data_frame_ids.append(line_info[1])
            f.close()
        else:
            f = open(os.path.join(dataroot, 'test_ids.txt'))
            for line in f.readlines():
                line_info = line.strip().split()
                self.data_ids.append(line_info[0])
                self.data_frame_ids.append(line_info[1])
            f.close()
            f2 = open(os.path.join(dataroot, 'test_cloth_ids.txt'))
            # f2 = open(os.path.join(dataroot, 'test_mvg_cloth_ids.txt'))
            for line in f2.readlines():
                line_info = line.strip().split()
                self.cloth_ids.append(line_info[0])
                self.cloth_frame_ids.append(line_info[1])
            f2.close()

        self.is_train = is_train
        self.sample_size = sample_size
        self.multi_length = multi_length
        self.clip_image_processor = CLIPProcessor.from_pretrained(clip_model_path,local_files_only=True)

        self.pixel_transforms = transforms.Compose([
            #transforms.Resize((1024,768), interpolation=0),
            #transforms.CenterCrop((int(1024 * 6/8), int(768 * 6/8))),
            transforms.Resize(self.sample_size, interpolation=0),
            # transforms.CenterCrop(self.sample_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.pixel_transforms_0 = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.pixel_transforms_1 = transforms.Compose([
            # transforms.Resize((1024,768), interpolation=0),
            # transforms.CenterCrop((int(1024 * 6/8), int(768 * 6/8))),
            transforms.Resize(self.sample_size, interpolation=0),
        ])

        self.ref_transforms_train = transforms.Compose([
            transforms.Resize(self.sample_size),
            # RandomScaleResize([1.0,1.1]),
            # transforms.CenterCrop(self.sample_size),
            transforms.RandomAffine(degrees=0, translate=(0.08,0.08),scale=(0.9,1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.ref_transforms_test = transforms.Compose([
            transforms.Resize(self.sample_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.output_front = True

    def __len__(self):
        if len(self.cloth_ids) >= 1:
            return len(self.data_ids)*len(self.cloth_ids)
        else:
            return len(self.data_ids)

    def __getitem__(self, idx):

        if len(self.cloth_ids) >=1:
            data_idx = idx // len(self.cloth_ids)
            cloth_idx = idx % len(self.cloth_ids)

            data_id = self.data_ids[data_idx]
            frame_id = self.data_frame_ids[data_idx]
            cloth_id = self.cloth_ids[cloth_idx]
            cloth_frame_id = self.cloth_frame_ids[cloth_idx]
            cloth_name_front = os.path.join(self.cloth_root, '%s_%s_front.jpg' % (cloth_id, cloth_frame_id))  # 实际是反的
            cloth_name_back = os.path.join(self.cloth_root, '%s_%s_back.jpg' % (cloth_id, cloth_frame_id))   
        else:
            data_id = self.data_ids[idx]
            frame_id = self.data_frame_ids[idx]
            cloth_name_front = os.path.join(self.cloth_root, '%s_%s_front.jpg' % (data_id, frame_id))  # 实际是反的
            cloth_name_back = os.path.join(self.cloth_root, '%s_%s_back.jpg' % (data_id, frame_id))

        # cloth_name_front = os.path.join(self.cloth_root, '%s_%s_front.jpg' % ('100030', '0540'))
        # cloth_name_back = os.path.join(self.cloth_root, '%s_%s_back.jpg' % ('100030', '0540'))

        images_root = os.path.join(self.dataroot, data_id, 'agnostic', frame_id)
        images = sorted(os.listdir(images_root))

        if self.is_train:
            check_images = []
            for image in images:
                if 'CC32871A015' not in image:
                    check_images.append(image)
            select_images = random.sample(check_images, self.multi_length)
            
        else:
            # front
            front_cameras = [
                'CC32871A005','CC32871A016','CC32871A017','CC32871A023','CC32871A027',
                'CC32871A030','CC32871A032','CC32871A033','CC32871A034','CC32871A035',
                'CC32871A038','CC32871A050','CC32871A051','CC32871A052','CC32871A059', 'CC32871A060'
            ]
            back_cameras = [
                'CC32871A004','CC32871A010', 'CC32871A013', 'CC32871A022', 'CC32871A029',
                'CC32871A031','CC32871A037', 'CC32871A039', 'CC32871A040', 'CC32871A044',
                'CC32871A046','CC32871A048', 'CC32871A055', 'CC32871A057', 'CC32871A058', 'CC32871A041'
            ]
            select_images = []
            for image in images:
                camera_id = image.split('_')[0]
                if camera_id in front_cameras and self.output_front:
                    select_images.append(image)
                if camera_id in back_cameras and not self.output_front:
                    select_images.append(image)         
        select_images = sorted(select_images)
        # print(select_images)
        for i in range(len(select_images)):
            select_images[i] = os.path.join(data_id,'resized_img', frame_id, select_images[i])
        sample = self.load_images(select_images, data_id, cloth_name_front, cloth_name_back)
        return sample
 
    def load_images(self, select_images, data_id, cloth_name_front, cloth_name_back):

        pixel_values_list = []
        pixel_values_pose_list = []
        camera_parm_list = []
        pixel_values_agnostic_list = []
        image_name_list = []

        # load camera info
        intri_name = os.path.join(self.dataroot, data_id, 'camera_intrinsics.json')
        extri_name = os.path.join(self.dataroot, data_id, 'camera_extrinsics.json')
        camera_scale_fn = os.path.join(self.dataroot, data_id, 'camera_scale.pkl')
        camera_scale = pickle.load(open(camera_scale_fn, "rb"))
        cameras_gt = read_camera_mvhumannet(intri_name, extri_name, camera_scale)

        # load person data
        for img_name in select_images:
            camera_id = img_name.split('/')[-1].split('_')[0]

            # load data
            image_name_list.append(img_name)
            pixel_values = Image.open(os.path.join(self.dataroot, img_name))
            pixel_values_pose = Image.open(os.path.join(self.dataroot, img_name).replace('resized_img', 'normals').replace('.jpg','_normal.jpg'))
            pixel_values_agnostic = Image.open(os.path.join(self.dataroot, img_name).replace('resized_img', 'agnostic'))
            parm_matrix = cameras_gt[camera_id]['RT']  # extrinsic

            # crop pose
            annot_path = os.path.join(self.dataroot, img_name.replace('resized_img', 'annots').replace('.jpg','.json'))
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
            pixel_values_pose = crop_and_resize(pixel_values_pose, bbox, size=self.sample_size)

            # camera parameter
            parm_matrix = torch.tensor(parm_matrix)
            camera_parm = parm_matrix[:3,:3].reshape(-1) # todo

            # transform
            pixel_values = self.pixel_transforms(pixel_values)
            pixel_values_pose = self.pixel_transforms(pixel_values_pose)
            pixel_values_agnostic = self.pixel_transforms(pixel_values_agnostic)
            
            pixel_values_list.append(pixel_values)
            pixel_values_pose_list.append(pixel_values_pose)
            camera_parm_list.append(camera_parm)
            pixel_values_agnostic_list.append(pixel_values_agnostic)

        pixel_values = torch.stack(pixel_values_list)
        pixel_values_pose = torch.stack(pixel_values_pose_list)
        camera_parm = torch.stack(camera_parm_list)
        pixel_values_agnostic = torch.stack(pixel_values_agnostic_list)

        pixel_values_cloth_front = Image.open(os.path.join(self.cloth_root, cloth_name_front))
        pixel_values_cloth_back = Image.open(os.path.join(self.cloth_root, cloth_name_back))

        # clip
        clip_ref_front = self.clip_image_processor(images=pixel_values_cloth_front, return_tensors="pt").pixel_values
        clip_ref_back = self.clip_image_processor(images=pixel_values_cloth_back, return_tensors="pt").pixel_values
        
        if self.is_train:
            pixel_values_cloth_front = self.ref_transforms_train(pixel_values_cloth_front)
            pixel_values_cloth_back = self.ref_transforms_train(pixel_values_cloth_back)
        else:
            pixel_values_cloth_front = self.ref_transforms_test(pixel_values_cloth_front)
            pixel_values_cloth_back = self.ref_transforms_test(pixel_values_cloth_back)

        drop_image_embeds = []
        for k in range(len(select_images)):
            if random.random() < 0.1:
                drop_image_embeds.append(torch.tensor(1))
            else:
                drop_image_embeds.append(torch.tensor(0))
        drop_image_embeds = torch.stack(drop_image_embeds)
        sample = dict(
            pixel_values=pixel_values, 
            pixel_values_pose=pixel_values_pose,
            pixel_values_agnostic=pixel_values_agnostic,
            clip_ref_front=clip_ref_front,
            clip_ref_back=clip_ref_back,
            pixel_values_cloth_front=pixel_values_cloth_front,
            pixel_values_cloth_back=pixel_values_cloth_back,
            camera_parm=camera_parm,
            drop_image_embeds=drop_image_embeds,
            img_name=image_name_list,
            cloth_name=cloth_name_front,
            )
        
        return sample

def collate_fn(data):
    
    pixel_values = torch.stack([example["pixel_values"] for example in data])
    pixel_values_pose = torch.stack([example["pixel_values_pose"] for example in data])
    pixel_values_agnostic = torch.stack([example["pixel_values_agnostic"] for example in data])
    clip_ref_front = torch.cat([example["clip_ref_front"] for example in data])
    clip_ref_back = torch.cat([example["clip_ref_back"] for example in data])
    pixel_values_cloth_front = torch.stack([example["pixel_values_cloth_front"] for example in data])
    pixel_values_cloth_back = torch.stack([example["pixel_values_cloth_back"] for example in data])
    camera_parm = torch.stack([example["camera_parm"] for example in data])
    drop_image_embeds = [example["drop_image_embeds"] for example in data]
    drop_image_embeds = torch.stack(drop_image_embeds)
    img_name = []
    cloth_name = []
    for example in data:
        img_name.extend(example['img_name'])
        cloth_name.append(example['cloth_name'])
    
    return {
        "pixel_values": pixel_values,
        "pixel_values_pose": pixel_values_pose,
        "pixel_values_agnostic": pixel_values_agnostic,
        "clip_ref_front": clip_ref_front,
        "clip_ref_back": clip_ref_back,
        "pixel_values_ref_front": pixel_values_cloth_front,
        "pixel_values_ref_back": pixel_values_cloth_back,
        "camera_parm": camera_parm,
        "drop_image_embeds": drop_image_embeds,
        "img_name": img_name,
        "cloth_name": cloth_name,
    }


if __name__ == '__main__':
    seed = 20
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = MVHumanNet_Dataset(dataroot="/GPUFS/sysu_gbli2_1/hzj/mvhumannet/",
        sample_size=(768,576),is_train=True,mode='pair',
        clip_model_path = "/GPUFS/sysu_gbli2_1/hzj/pretrained_models/clip-vit-base-patch32")

    # print(len(dataset))

    # for _ in range(500):

    #     p = random.randint(0,len(dataset)-1)
    #     p = dataset[p]

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=2,
    )

    for _, batch in enumerate(test_dataloader):
        # print(batch['cloth_name'], batch['img_name'])
        p = {}
        print('111', batch['camera_parm'].shape)
        print('111', batch['drop_image_embeds'].shape)
        for key in batch.keys():
            p[key] = batch[key][0]
        # p = dataset[12]

        print(p['camera_parm'].shape)

        pixel_values = p['pixel_values'][0].permute(1,2,0).numpy()
        print(p['pixel_values'].shape)
        pixel_values = pixel_values / 2 + 0.5
        pixel_values *=255
        pixel_values = pixel_values.astype(np.uint8)
        pixel_values= Image.fromarray(pixel_values)
        pixel_values.save('pixel_values0.jpg')

        pixel_values_pose = p['pixel_values_pose'][0].permute(1,2,0).numpy()
        print(p['pixel_values_pose'].shape)
        pixel_values_pose = pixel_values_pose / 2 + 0.5
        pixel_values_pose *=255
        pixel_values_pose = pixel_values_pose.astype(np.uint8)
        pixel_values_pose= Image.fromarray(pixel_values_pose)
        pixel_values_pose.save('pixel_values_pose.jpg')

        pixel_values_agnostic = p['pixel_values_agnostic'][0].permute(1,2,0).numpy()
        print(p['pixel_values_agnostic'].shape)
        pixel_values_agnostic = pixel_values_agnostic / 2 + 0.5
        pixel_values_agnostic *=255
        pixel_values_agnostic = pixel_values_agnostic.astype(np.uint8)
        pixel_values_agnostic= Image.fromarray(pixel_values_agnostic)
        pixel_values_agnostic.save('pixel_values_agnostic.jpg')

        pixel_values = p['pixel_values'][2].permute(1,2,0).numpy()
        print(p['pixel_values'].shape)
        pixel_values = pixel_values / 2 + 0.5
        pixel_values *=255
        pixel_values = pixel_values.astype(np.uint8)
        pixel_values= Image.fromarray(pixel_values)
        pixel_values.save('pixel_values2.jpg')

        pixel_values_pose = p['pixel_values_pose'][2].permute(1,2,0).numpy()
        print(p['pixel_values_pose'].shape)
        pixel_values_pose = pixel_values_pose / 2 + 0.5
        pixel_values_pose *=255
        pixel_values_pose = pixel_values_pose.astype(np.uint8)
        pixel_values_pose= Image.fromarray(pixel_values_pose)
        pixel_values_pose.save('pixel_values_pose2.jpg')

        pixel_values_agnostic = p['pixel_values_agnostic'][2].permute(1,2,0).numpy()
        print(p['pixel_values_agnostic'].shape)
        pixel_values_agnostic = pixel_values_agnostic / 2 + 0.5
        pixel_values_agnostic *=255
        pixel_values_agnostic = pixel_values_agnostic.astype(np.uint8)
        pixel_values_agnostic= Image.fromarray(pixel_values_agnostic)
        pixel_values_agnostic.save('pixel_values_agnostic2.jpg')

        pixel_values_cloth_img = p['pixel_values_ref_front'].permute(1,2,0).numpy()
        print(p['pixel_values_ref_front'].shape)
        pixel_values_cloth_img = pixel_values_cloth_img / 2 + 0.5
        pixel_values_cloth_img *=255
        pixel_values_cloth_img = pixel_values_cloth_img.astype(np.uint8)
        pixel_values_cloth_img= Image.fromarray(pixel_values_cloth_img)
        pixel_values_cloth_img.save('pixel_values_cloth_front.jpg')

        pixel_values_cloth_img = p['pixel_values_ref_back'].permute(1,2,0).numpy()
        print(p['pixel_values_ref_back'].shape)
        pixel_values_cloth_img = pixel_values_cloth_img / 2 + 0.5
        pixel_values_cloth_img *=255
        pixel_values_cloth_img = pixel_values_cloth_img.astype(np.uint8)
        pixel_values_cloth_img= Image.fromarray(pixel_values_cloth_img)
        pixel_values_cloth_img.save('pixel_values_cloth_back.jpg')
        exit()


