import torch
import numpy as np
from models.mv_attn_processor import MVXFormersAttnProcessor

def cal_cos_dist(R1, R2):
    # 假定 R1 和 R2 都是 3x3 的旋转矩阵  

    # 计算相对旋转矩阵 R_rel
    R_rel = torch.matmul(R1.transpose(-2, -1), R2) 

    # 计算 R_rel 的迹，并计算余弦值
    trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1) / 2 + 1) / 2
    return cos_theta

def calculate_weight_matrix(camera_pose):
    # 假定 camera_pose 是一个 (b, frame_length, 3, 3) 的张量

    frame_length = camera_pose.size(1)
    weight_matrix = torch.zeros(camera_pose.size(0), frame_length, frame_length)

    for i in range(frame_length):
        R1 = camera_pose[:, i]
        for j in range(frame_length):
            R2 = camera_pose[:, j]
            weight_matrix[:, i, j] = cal_cos_dist(R1, R2)

    return weight_matrix

def register_mv_attn(unet):
    attn_procs = {}
    for i, name in enumerate(unet.attn_processors.keys()):
        is_self_attn = (i % 2 == 0)

        if is_self_attn:
            attn_procs[name] = MVXFormersAttnProcessor()
        else:
            attn_procs[name] = unet.attn_processors[name]

    unet.set_attn_processor(attn_procs)

def update_mv_attn(unet, weight_matrix):
    for i, name in enumerate(unet.attn_processors.keys()):
        is_self_attn = (i % 2 == 0)

        if is_self_attn:
            unet.attn_processors[name].update_weight_matrix(weight_matrix)

def camera_para_embed(camera_pose, embed_dim=64):
    """
    计算相机参数的编码。

    参数:
        camera_pose (Tensor): 形状为 (batch_size, rotation_matrix_dim) 的张量。
        embed_dim (int): 嵌入维度。

    返回:
        Tensor: 形状为 (batch_size, rotation_matrix_dim * embed_dim * 2) 的编码张量。
    """
    batch_size = camera_pose.shape[0]
    rotation_matrix_dim = camera_pose.shape[1]
    
    # 创建角度的幂次指数，计算出 a1
    a1 = torch.arange(embed_dim, device=camera_pose.device)
    a1 = torch.pow(2, a1) * torch.pi  # (embed_dim)
    
    # 计算每个姿态的编码
    # Expand `a1` to match the shape of `camera_pose` for broadcasting
    a1_expanded = a1.unsqueeze(0).unsqueeze(0)  # (1, 1, embed_dim)
    
    # Expand `camera_pose` to include embedding dimensions for broadcasting
    camera_pose_expanded = camera_pose.unsqueeze(-1)  # (batch_size, rotation_matrix_dim, 1)
    
    # Compute sin and cos for all dimensions at once
    t1 = torch.sin(camera_pose_expanded * a1_expanded)  # (batch_size, rotation_matrix_dim, embed_dim)
    t2 = torch.cos(camera_pose_expanded * a1_expanded)  # (batch_size, rotation_matrix_dim, embed_dim)
    
    # Reshape tensors to the desired output shape
    camera_embed = torch.cat([t1, t2], dim=-1)  # (batch_size, rotation_matrix_dim, embed_dim * 2)
    
    # Reshape to the final output shape
    camera_embed = camera_embed.view(batch_size, -1)  # (batch_size, rotation_matrix_dim * embed_dim * 2)
    
    return camera_embed