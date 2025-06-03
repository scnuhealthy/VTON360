import numpy as np
import os
import json

# 定义反向的 tf_mat2extr 函数 (C2W -> W2C)
def tf_mat2extr(tf_mat):
    """
    Convert a camera-to-world (C2W) transformation matrix back to
    a world-to-camera (W2C) extrinsic matrix.
    """
    tf_mat_inv = np.linalg.inv(tf_mat)

    trans_mat = np.eye(4)
    trans_mat[1, 1] = -1
    trans_mat[2, 2] = -1

    tf_mat_inv = trans_mat @ tf_mat_inv

    extr = np.stack([tf_mat_inv[0, :], tf_mat_inv[1, :], tf_mat_inv[2, :]], axis=0)
    
    return extr[:, :4]

root = '/data1/hezijian/save_render_data_yw/all/'   # revise your path here

for sub_folder in os.listdir(root):
    # sub_folder = '0000'
    sub_folder_json_path = os.path.join(root, sub_folder, 'transforms.json')
    params_path = os.path.join(root,sub_folder,'parm')
    info = json.load(open(sub_folder_json_path))
    frames = info['frames']
    if not os.path.exists(params_path):
        os.makedirs(params_path)
    for frame in frames:
        transform_matrix = frame['transform_matrix']
        flx = frame['fl_x']
        fly = frame['fl_y']
        cx = frame['cx']
        cy = frame['cy']
        file_name = frame['file_path'].split('/')[1].split('.')[0]

        transform_matrix = np.array(transform_matrix)
        extrinsic_matrix = tf_mat2extr(transform_matrix)
        intrinsic_matrix = np.array([[flx, 0, cx],[0,fly,cy],[0,0,1]])
        np.save(os.path.join(params_path,'%s_extrinsic.npy'%file_name),extrinsic_matrix)
        np.save(os.path.join(params_path,'%s_intrinsic.npy'%file_name),intrinsic_matrix)
    