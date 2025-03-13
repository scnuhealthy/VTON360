import taichi_three as t3
import numpy as np
from taichi_three.transform import *
import math
from pathlib import Path
from tqdm import tqdm
import os
import cv2
import pickle
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def extr2tf_mat(extr):
    """
    Return C2W matrix in the shape of (4, 4).
    """
    # flip y and z axis in the camera coordinate space, according to
    # [ns-data conventions](https://docs.nerf.studio/quickstart/data_conventions.html#camera-view-space)
    trans_mat = np.eye(4)
    trans_mat[1, 1] = -1
    trans_mat[2, 2] = -1

    tf_mat = np.stack([extr[0], extr[1], extr[2], [0, 0, 0, 1]], axis=0)
    tf_mat = trans_mat @ tf_mat
    tf_mat = np.linalg.inv(tf_mat)
    return tf_mat


def intr2dict(intr: np.ndarray):
    intr_dict = {
        "fl_x": intr[0, 0],
        "fl_y": intr[1, 1],
        "cx": intr[0, 2],
        "cy": intr[1, 2],
    }
    return intr_dict


def save(data_id, save_path, extrs, intrs, depths, imgs, masks):
    instance_path = os.path.join(save_path, data_id)
    img_save_path = os.path.join(save_path, data_id, 'images')
    depth_save_path = os.path.join(save_path, data_id, 'depth')
    mask_save_path = os.path.join(save_path, data_id, 'mask')
    Path(img_save_path).mkdir(exist_ok=True, parents=True)
    Path(mask_save_path).mkdir(exist_ok=True, parents=True)
    Path(depth_save_path).mkdir(exist_ok=True, parents=True)

    h, w = imgs[0].shape[:2]
    transforms_dict = {
        "w": w,
        "h": h,
        "k1": 0, "k2": 0, "p1": 0, "p2": 0,
        "camera_model": "OPENCV",
    }

    frames = []
    num_views = len(extrs)
    for pid in range(num_views):
        # nyw note: depth is normalized to [0, 1] in taichi_three，so we need to scale it back to [0, 2^15]
        depth = depths[pid] * 2.0 ** 15
        cv2.imwrite(os.path.join(depth_save_path, f'{pid:03d}.png'), depth.astype(np.uint16))

        img = (np.clip(imgs[pid], 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(img_save_path, f'{pid:03d}.jpg'), img)

        mask = (np.clip(masks[pid], 0, 1) * 255.0 + 0.5).astype(np.uint8)
        cv2.imwrite(os.path.join(mask_save_path, f'{pid:03d}.png'), mask[:, :, 0])

        frame_dict = {
            "file_path": f"images/{pid:03d}.jpg",
            "mask_path": f"mask/{pid:03d}.png",
            "depth_file_path": f"depth/{pid:03d}.png",
            "transform_matrix": extr2tf_mat(extrs[pid]).tolist(),
        }
        intr_dict = intr2dict(intrs[pid])
        frame_dict.update(intr_dict)
        frames.append(frame_dict)
    transforms_dict["frames"] = frames

    with open(os.path.join(instance_path, "transforms.json"), "w") as f:
        json.dump(transforms_dict, f, indent=4)
    # with open(os.path.join(instance_path, "_transforms.json"), "w") as f:
    #     json.dump(transforms_dict, f, indent=4)

def save_normal(data_id, save_path, depths, normals):
    normal_save_path = os.path.join(save_path, data_id, 'normals')
    depth_save_path = os.path.join(save_path, data_id, 'depth_smpls')
    Path(normal_save_path).mkdir(exist_ok=True, parents=True)
    Path(depth_save_path).mkdir(exist_ok=True, parents=True)

    num_views = len(depths)
    for pid in range(num_views):
        # nyw note: depth is normalized to [0, 1] in taichi_three，so we need to scale it back to [0, 2^15]
        depth = depths[pid] * 2.0 ** 15
        cv2.imwrite(os.path.join(depth_save_path, f'{pid:03d}.png'), depth.astype(np.uint16))

        normal = (np.clip(normals[pid], 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(normal_save_path, f'{pid:03d}.jpg'), normal)



class StaticRenderer:
    def __init__(self):
        ti.init(arch=ti.cuda, device_memory_fraction=0.8)
        self.scene = t3.Scene()
        self.N = 10
    
    def change_all(self):
        save_obj = []
        save_tex = []
        for model in self.scene.models:
            save_obj.append(model.init_obj)
            save_tex.append(model.init_tex)
        ti.init(arch=ti.cuda, device_memory_fraction=0.8)
        print('init')
        self.scene = t3.Scene()
        for i in range(len(save_obj)):
            model = t3.StaticModel(self.N, obj=save_obj[i], tex=save_tex[i])
            self.scene.add_model(model)

    def check_update(self, obj):
        temp_n = self.N
        self.N = max(obj['vi'].shape[0], self.N)
        self.N = max(obj['f'].shape[0], self.N)
        if not (obj['vt'] is None):
            self.N = max(obj['vt'].shape[0], self.N)

        if self.N > temp_n:
            self.N *= 2
            self.change_all()
            self.camera_light()
    
    def add_model(self, obj, tex=None):
        self.check_update(obj)
        model = t3.StaticModel(self.N, obj=obj, tex=tex)
        self.scene.add_model(model)
    
    def modify_model(self, index, obj, tex=None):
        self.check_update(obj)
        self.scene.models[index].init_obj = obj
        self.scene.models[index].init_tex = tex
        self.scene.models[index]._init()
    
    def camera_light(self):
        camera = t3.Camera(res=(1024, 1024))
        self.scene.add_camera(camera)

        camera_hr = t3.Camera(res=(2048, 2048))
        self.scene.add_camera(camera_hr)
        
        light_dir = np.array([0, 0, 1])
        light_list = []
        for l in range(6):
            rotate = np.matmul(rotationX(math.radians(np.random.uniform(-30, 30))),
                               rotationY(math.radians(360 // 6 * l)))
            dir = [*np.matmul(rotate, light_dir)]
            light = t3.Light(dir, color=[1.0, 1.0, 1.0])
            light_list.append(light)
        lights = t3.Lights(light_list)
        self.scene.add_lights(lights)


def render_data(renderer, data_path, phase, data_id, save_path, cam_nums, res, dis=1.0, is_thuman=False, is_smpl_model=False, seed_value=0):
    np.random.seed(seed_value)
    if not is_smpl_model:
        obj_path = os.path.join(data_path, phase, data_id, '%s.obj' % data_id)
        texture_path = data_path
        img_path = os.path.join(texture_path, phase, data_id, 'material0.jpeg')
        texture = cv2.imread(img_path)[:, :, ::-1]

        # ################ nyw add equalizeHist for texture ################
        # # comment out the following lines to disable equalizeHist for texture
        # texture = cv2.cvtColor(texture, cv2.COLOR_RGB2HSV)
        # texture[:, :, 2] = cv2.equalizeHist(texture[:, :, 2]) * 0.85 # scale down the brightness by 0.85
        # texture = cv2.cvtColor(texture, cv2.COLOR_HSV2RGB)
        # ################ nyw add equalizeHist for texture ################
    
        texture = np.ascontiguousarray(texture)
        texture = texture.swapaxes(0, 1)[:, ::-1, :]
    else:
        # obj_path = '/data1/hezijian/Thuman2.1_GPS/0000.obj'
        obj_path = '/data1/hezijian/Thuman2.1/THuman2.0_Smpl_X_Paras/%s/mesh_smplx.obj' % data_id

    obj = t3.readobj(obj_path, scale=1)

    # height normalization
    vy_max = np.max(obj['vi'][:, 1])
    vy_min = np.min(obj['vi'][:, 1])
    human_height = 1.80 + np.random.uniform(-0.05, 0.05, 1)
    obj['vi'][:, :3] = obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    obj['vi'][:, 1] -= np.min(obj['vi'][:, 1])
    look_at_center = np.array([0, 0.85, 0]) 
    base_cam_pitch = -8

    ################ nyw: add multi-pitchs for better reconstruction ################
    # base_cam_pitch = -8
    cam_pitchs = [-8, 45, -45, 90, -90]
    cam_nums_for_each_pitch = [cam_nums, cam_nums//2, cam_nums//2, 1, 1]
    ################ nyw: add multi-pitchs for better reconstruction ################

    # randomly move the scan
    move_range = 0.1 if human_height < 1.80 else 0.05
    delta_x = np.max(obj['vi'][:, 0]) - np.min(obj['vi'][:, 0])
    delta_z = np.max(obj['vi'][:, 2]) - np.min(obj['vi'][:, 2])
    if delta_x > 1.0 or delta_z > 1.0:
        move_range = 0.01
    obj['vi'][:, 0] += np.random.uniform(-move_range, move_range, 1)
    obj['vi'][:, 2] += np.random.uniform(-move_range, move_range, 1)

    if len(renderer.scene.models) >= 1:
        if not is_smpl_model:
            renderer.modify_model(0, obj, texture)
        else:
            renderer.modify_model(0, obj)
    else:
        if not is_smpl_model:
            renderer.add_model(obj, texture)
        else:
            renderer.add_model(obj)

    if is_thuman:
        # thuman needs a normalization of orientation
        smpl_path = os.path.join(data_path, 'THuman2.0_Smpl_X_Paras', data_id, 'smplx_param.pkl')
        with open(smpl_path, 'rb') as f:
            smpl_para = pickle.load(f)

        y_orient = smpl_para['global_orient'][0][1]  
        angle_base = (y_orient*180.0/np.pi)

    # nyw note: generate one instance of thuman in this loop
    extrs, intrs, depths, imgs, masks, normals = [], [], [], [], [], []
    for ci, cam_pitch in enumerate(cam_pitchs):
      for pid in range(cam_nums := cam_nums_for_each_pitch[ci]):
        degree_interval = 360 / cam_nums
        angle = angle_base + pid * degree_interval

        def render(dis, angle, look_at_center, p, renderer, render_2k=False, render_normal=False):
            ori_vec = np.array([0, 0, dis])
            rotate = np.matmul(rotationY(math.radians(angle)), rotationX(math.radians(p)))
            fwd = np.matmul(rotate, ori_vec)
            cam_pos = look_at_center + fwd

            x_min = 0
            y_min = -25
            cx = res[0] * 0.5
            cy = res[1] * 0.5
            fx = res[0] * 0.8
            fy = res[1] * 0.8
            _cx = cx - x_min
            _cy = cy - y_min
            renderer.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
            renderer.scene.cameras[0].set(pos=cam_pos, target=look_at_center)
            renderer.scene.cameras[0]._init()

            if render_2k:
                fx = res[0] * 0.8 * 2
                fy = res[1] * 0.8 * 2
                _cx = (res[0] * 0.5 - x_min) * 2
                _cy = (res[1] * 0.5 - y_min) * 2
                renderer.scene.cameras[1].set_intrinsic(fx, fy, _cx, _cy)
                renderer.scene.cameras[1].set(pos=cam_pos, target=look_at_center)
                renderer.scene.cameras[1]._init()

                renderer.scene.render()
                camera = renderer.scene.cameras[0]
                camera_hr = renderer.scene.cameras[1]
                extrinsic = camera.export_extrinsic()
                intrinsic = camera.export_intrinsic()
                depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
                img = camera.img.to_numpy().swapaxes(0, 1)
                img_hr = camera_hr.img.to_numpy().swapaxes(0, 1)
                mask = camera.mask.to_numpy().swapaxes(0, 1)
                return extrinsic, intrinsic, depth_map, img, mask, img_hr 
                
            renderer.scene.render()
            camera = renderer.scene.cameras[0]
            extrinsic = camera.export_extrinsic()
            intrinsic = camera.export_intrinsic()
            depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
            if not render_normal:
                img = camera.img.to_numpy().swapaxes(0, 1)
            else:
                img = camera.normal_map.to_numpy().swapaxes(0, 1)
            mask = camera.mask.to_numpy().swapaxes(0, 1)
            return extrinsic, intrinsic, depth_map, img, mask

        if not is_smpl_model:
            extr, intr, depth, img, mask = render(dis, angle, look_at_center, cam_pitch, renderer)
            extrs.append(extr)
            intrs.append(intr)
            depths.append(depth)
            imgs.append(img)
            masks.append(mask)
        else:
            extr, intr, depth, img, mask = render(dis, angle, look_at_center, cam_pitch, renderer, render_normal=True)
            depths.append(depth)
            normals.append(img)

    if not is_smpl_model:
        save(data_id, save_path, extrs, intrs, depths, imgs, masks)
    else:
        save_normal(data_id, save_path, depths, normals)

if __name__ == '__main__':
    cam_nums = 80
    scene_radius = 2.0
    res = (1024, 1024)
    thuman_root = '/PATH/TO/Thuman2.1/'
    save_root = '/PATH/TO/OUTPUT'
    renderer = StaticRenderer()

    # for phase in ['train', 'val']:
    phase = 'all'
    thuman_list = sorted(os.listdir(os.path.join(thuman_root, phase)))
    thuman_list = thuman_list[512:513]
    save_path = os.path.join(save_root, phase)
    seed_value = np.random.randint(1,1000)
    for data_id in tqdm(thuman_list):
        render_data(renderer, thuman_root, phase, data_id, save_path, cam_nums, res, dis=scene_radius, is_thuman=True, seed_value=seed_value)
        render_data(renderer, thuman_root, phase, data_id, save_path, cam_nums, res, dis=scene_radius, is_thuman=True, is_smpl_model=True, seed_value=seed_value)