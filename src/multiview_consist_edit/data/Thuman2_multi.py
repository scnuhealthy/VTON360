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

def crop_image(human_img_orig):
    human_img_orig = human_img_orig.resize((1024,1024))
    original_width, original_height = human_img_orig.size
    target_width = 768
    crop_amount = (original_width - target_width) // 2
    left = crop_amount
    upper = 0
    right = original_width - crop_amount
    lower = original_height
    cropped_image = human_img_orig.crop((left, upper, right, lower))
    return cropped_image

class Thuman2_Dataset(Dataset):
    def __init__(
        self, dataroot, sample_size=(512,384), is_train=True, mode='pair', clip_model_path='', multi_length=8, output_front=True,
    ):
        c_names_front = []
        c_names_back = []

        self.data_ids = []
        self.dataroot = os.path.join(dataroot, 'all')
        self.cloth_root = os.path.join(dataroot, 'cloth')
        # self.cloth_root = os.path.join(dataroot, 'MVG_clothes')

        self.cloth_ids = []
        if is_train:
            f = open(os.path.join(dataroot,'train_ids.txt'))
            for line in f.readlines():
                self.data_ids.append(line.strip())
            f.close()
        else:
            # f = open(os.path.join(dataroot, 'val_ids.txt'))
            f = open(os.path.join(dataroot, 'test_ids.txt'))
            # f = open(os.path.join(dataroot, 'test_mvg_ids.txt'))
            for line in f.readlines():
                self.data_ids.append(line.strip())
            f.close()
            f2 = open(os.path.join(dataroot, 'test_cloth_ids.txt'))
            # f2 = open(os.path.join(dataroot, 'test_mvg_cloth_ids.txt'))
            for line in f2.readlines():
                self.cloth_ids.append(line.strip())
            f2.close()

        self.mode = mode
        self.is_train = is_train
        self.sample_size = sample_size
        self.multi_length = multi_length
        self.clip_image_processor = CLIPProcessor.from_pretrained(clip_model_path,local_files_only=True)

        self.pixel_transforms = transforms.Compose([
            transforms.Resize((1024,768), interpolation=0),
            transforms.CenterCrop((int(1024 * 6/8), int(768 * 6/8))),
            transforms.Resize(self.sample_size, interpolation=0),
            # transforms.CenterCrop(self.sample_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.pixel_transforms_0 = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.pixel_transforms_1 = transforms.Compose([
            transforms.Resize((1024,768), interpolation=0),
            transforms.CenterCrop((int(1024 * 6/8), int(768 * 6/8))),
            transforms.Resize(self.sample_size, interpolation=0),
        ])

        self.ref_transforms_train = transforms.Compose([
            transforms.Resize(self.sample_size),
            # RandomScaleResize([1.0,1.1]),
            transforms.CenterCrop(self.sample_size),
            transforms.RandomAffine(degrees=0, translate=(0.08,0.08),scale=(0.9,1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.ref_transforms_test = transforms.Compose([
            transforms.Resize(self.sample_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.color_transform = transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.0)
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
            cloth_id = self.cloth_ids[cloth_idx]
            cloth_name_back = os.path.join(self.cloth_root, '%s_front.jpg' % cloth_id)
            cloth_name_front =  os.path.join(self.cloth_root, '%s_back.jpg' % cloth_id)       
        else:
            data_id = self.data_ids[idx]
            cloth_name_back = os.path.join(self.cloth_root, '%s_front.jpg' % data_id)
            cloth_name_front =  os.path.join(self.cloth_root, '%s_back.jpg' % data_id)

        images_root = os.path.join(self.dataroot, data_id, 'agnostic') # need only val
        images = sorted(os.listdir(images_root))

        # cloth_name_back = '0001_front.jpg'
        # cloth_name_front = '0001_back.jpg'

        if self.is_train:
            select_images = random.sample(images, self.multi_length)
            
        else:
            # select_idxs = [0,3,6,9,12, 15,18,21,24,27, 79,76,73,70,67,64]
            L = len(images)
            select_idxs = []
            begin = 0
            sl = 16.0
            if self.output_front:
                while begin < L//2:
                    select_idxs.append(int(begin/2))
                    select_idxs.append(int(L-1-begin/2))
                    begin += L/sl
            else:
                begin = L//4
                while begin < L*3//4:
                    select_idxs.append(int(begin))
                    begin += L/2/sl
            # print(sorted(select_idxs))
            # select_idxs = [0,3,6,9,12, 15,18,21,24,27, L-1,L-4,L-7,L-10,L-13,L-16]
            select_images = []
            for select_idx in select_idxs:
                select_images.append(images[select_idx])
        select_images = sorted(select_images)
        # print(select_images)
        for i in range(len(select_images)):
            select_images[i] = os.path.join(data_id,'images',select_images[i])
        sample = self.load_images(select_images, cloth_name_front, cloth_name_back)
        return sample

    def color_progress(images):
        fn_idx, b, c, s, h = self.color_transform.get_params(color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,color_jitter.hue)
        for image in images:    
            image = F.adjust_contrast(image, c)
            image = F.adjust_brightness(image, b)
            image = F.adjust_saturation(image, s)
        return images    

    def load_images(self, select_images, cloth_name_front, cloth_name_back):

        pixel_values_list = []
        pixel_values_pose_list = []
        camera_parm_list = []
        pixel_values_agnostic_list = []
        image_name_list = []

        # load person data
        for img_name in select_images:
            image_name_list.append(img_name)
            pixel_values = Image.open(os.path.join(self.dataroot, img_name))
            pixel_values_pose = Image.open(os.path.join(self.dataroot, img_name).replace('images', 'normals'))
            # parse_lip = Image.open(os.path.join(parse_lip_dir, img_name))
            pixel_values_agnostic = Image.open(os.path.join(self.dataroot, img_name).replace('images', 'agnostic'))
            parm_matrix = np.load(os.path.join(self.dataroot, img_name[:4],'parm', img_name[-7:-4]+'_extrinsic.npy'))
            pixel_values = crop_image(pixel_values)
            pixel_values_pose = crop_image(pixel_values_pose)
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
    dataset = Thuman2_Dataset(dataroot="/GPUFS/sysu_gbli2_1/hzj/save_render_data_yw/",
        sample_size=(768,576),is_train=False,mode='pair',
        clip_model_path = "/GPUFS/sysu_gbli2_1/hzj/pretrained_models/clip-vit-base-patch32")

    # for _ in range(500):

        # p = random.randint(0,len(dataset)-1)
        # p = dataset[p]

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=2,
        num_workers=1,
    )

    for _, batch in enumerate(test_dataloader):
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