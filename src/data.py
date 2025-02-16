"""
data.py

This module defines the functions for extracting the information from the RealEstate10k dataset and doing the necessary processing.

Authors: Sandro Mikautadze, Elio Samaha.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
from typing import List, Tuple, Optional
import random
from einops import rearrange, repeat

def _get_camera_params_for_file(filepath: str, w: int = 1, h: int = 1, device: torch.device = "cpu"):
    """
    Parses camera parameters from a given RealEstate10K file.

    Args:
        filepath (str): Path for the txt file.
        w (int): Width of the video. Default is 1 (normalized).
        h (int): Height of the video. Default is 1 (normalized).
        device (torch.device): Device for computation (CPU/GPU). Default is "cpu".

    Returns:
        tuple:
            - url (str): URL of the video.
            - K (torch.Tensor): 3x3 camera intrinsics matrix.
            - timestamps (np.ndarray): NumPy array of timestamps (in microseconds) for each frame.
            - extrinsics (torch.Tensor): [num_frames, 3, 4] Camera extrinsics matrices.
    """
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
    # lines is a list of str representing lines from a file, containing
    # the video URL, timestamps, camera intrinsics, and extrinsics.
    
    # video URL
    url = lines[0].strip()
        
    # camera intrinsics (done just on first frame since it is constant across time)
    # normalized coordinates: top left corner of the image is (0, 0) and bottom right corner is (1, 1)
    first_frame = lines[1].strip().split()
    f_x, f_y, c_x, c_y = map(float, first_frame[1:5])
    K = torch.tensor([[w * f_x, 0,       w * c_x],
                      [0,       h * f_y, h * c_y],
                      [0,       0,       1]], dtype=torch.float32, device=device)
    
    # camera extrinsics and time
    timestamps = []
    extrinsics  = []
    for line in lines[1:]: # skip the first line (video URL)
        frame = line.strip().split()
        timestamp = int(frame[0])
        timestamps.append(timestamp)
        P = torch.tensor(list(map(float, frame[7:])), dtype=torch.float32, device=device).reshape(3, 4)
        extrinsics.append(P)
    
    return url, K, torch.tensor(timestamps, dtype=torch.float32, device=device), torch.stack(extrinsics)

def _apply_distortion(u: torch.Tensor, v: torch.Tensor,
                     cx: float = 0.5, cy: float =0.5,
                     d1: float = 0.0, d2: float = 0.0, d3: float = 0.0):
    """
    Applies radial distortion to the standard pinhole camera model.

    Args:
        u (torch.Tensor): Normalized x-coordinates.
        v (torch.Tensor): Normalized y-coordinates.
        cx, cy (float): Principal point (center of distortion). Default is 0.5 (normalized).
        d1, d2, d3 (float): Radial distortion parameters.

    Returns:
        (u_D, v_D): Distorted coordinates.
    """
    
    r_squared = (u - cx) ** 2 + (v - cy) ** 2
    D = 1 + d1 * r_squared + d2 * r_squared**2 + d3 * r_squared**3
    
    return (u - cx) * D + cx, (v - cy) * D + cy

def _precompute_homogeneous_uv(num_rays_x: int = 16, num_rays_y: int = 9, 
                              distortion_params: Optional[Tuple[float, float, float]] = None,
                              device: torch.device = "cpu"):
    """
    Precomputes a homogeneous grid of (u,v,1) coordinates, optionally applying radial distortion.

    Args:
        num_rays_x (int): Number of horizontal rays. Default is 16.
        num_rays_y (int): Number of vertical rays. Default is 9.
        distortion_params (tuple or None): (d1, d2, d3) radial distortion parameters. Default to None (no distortion).
        device (torch.device): Device for computation (CPU/GPU). Default is "cpu".

    Returns:
        torch.Tensor: Homogeneous (u, v, 1) grid of shape [3, num_rays_x * num_rays_y].
    """
    u = (torch.arange(num_rays_x, device=device, dtype=torch.float32) + 0.5) / num_rays_x # ensure it's in the middle of the pixel
    v = (torch.arange(num_rays_y, device=device, dtype=torch.float32) + 0.5) / num_rays_y
    U, V = torch.meshgrid(u, v, indexing="ij")
    
    # Skip distortion if parameters are (0.0, 0.0, 0.0)
    if distortion_params is not None and not torch.allclose(torch.tensor(distortion_params, device=device), torch.zeros(3, device=device), atol=1e-7):
        U, V = _apply_distortion(U, V, d1=distortion_params[0],
                                          d2=distortion_params[1],
                                          d3=distortion_params[2])
    
    # returns final shape [3, num_rays] where num_rays = num_rays_x * num_rays_y 
    # 0 contains the x coordinates, 1 contains the y coordinates, 2 contains the z coordinates (all 1s)
    # so homogeneous_uv[:, i] is [u_i, v_i, 1]
    
    return torch.stack([U.flatten(), V.flatten(), torch.ones_like(U.flatten())], dim=0)


# NOT USED ANYMORE!!!!!!!!
# def get_plucker_embedding(K_inv: torch.Tensor, P: torch.Tensor,
#                           num_rays_x: int = 16, num_rays_y: int = 9,
#                           distortion_params: Optional[Tuple[float, float, float]] = None):
#     """
#     Computes Plücker ray embeddings (direction and moment) given camera intrinsics and extrinsics.

#     Args:
#         K_inv (torch.Tensor): 3x3 inverse intrinsic matrix.
#         P (torch.Tensor): 3x4 extrinsic matrix.
#         num_rays_x (int): Number of horizontal rays. Default is 16.
#         num_rays_y (int): Number of vertical rays. Default is 9.
#         distortion_params (tuple or None): (d1, d2, d3) radial distortion parameters. Default to None (no distortion).

#     Returns:
#         plucker_embeddings (torch.Tensor): [6, num_rays_y, num_rays_x] Plücker embeddings with direction and moment.
#     """
    
#     R, t = P[:, :3], P[:, 3]
    
#     u = np.linspace(0, 1, num_rays_x, endpoint=False) + 0.5 / num_rays_x # ensure it's in the middle of the pixel
#     v = np.linspace(0, 1, num_rays_y, endpoint=False) + 0.5 / num_rays_y 
#     U, V = np.meshgrid(u, v)
    
#     ## distortion
#     if distortion_params is None:
#         u_dist, v_dist = torch.tensor(U, dtype=torch.float32, device=K_inv.device), torch.tensor(V, dtype=torch.float32, device=K_inv.device)
#     else: 
#         u_dist, v_dist = apply_distortion(torch.tensor(U, dtype=torch.float32, device=K_inv.device),
#                                           torch.tensor(V, dtype=torch.float32, device=K_inv.device),
#                                           d1=distortion_params[0], d2=distortion_params[1], d3=distortion_params[2])
    
#     homogeneous_uv = torch.stack([u_dist.flatten(), v_dist.flatten(), torch.ones_like(u_dist.flatten())], dim=0) # [3, num_rays] where num_rays = num_rays_x * num_rays_y 
#     # 0 contains the x coordinates, 1 contains the y coordinates, 2 contains the z coordinates (all 1s)
#     # so homogeneous_uv[:, i] is [u_i, v_i, 1]
    
#     # ray direction d = R^T * K^-1 * [u, v, 1]^T
#     d = R.T @ K_inv @ homogeneous_uv # [3, num rays]
#     d = d / torch.norm(d, dim=0, keepdim=True) 
#     d = d.T # [num_rays, 3] (each row is a direction vector)
    
#     # ray origins (camera center in world coordinate)
#     camera_center = - R.T @ t # [3]
#     O = camera_center.expand(d.shape[0], -1) # [num_rays, 3]
#     # ray moment m = o x d
#     m = torch.cross(O, d, dim=1) # [num_rays, 3], PS: since ||d||=1, then ||m||=distance from ray to origin
    
#     plucker_embedding = torch.cat([d, m], dim = 1) # [num_rays, 6]
#     plucker_embedding = rearrange(plucker_embedding, "(h w) c -> c h w", h=num_rays_y, w=num_rays_x) 
    
#     return plucker_embedding

def parse_directory(directory: str,
                    w: int = 1, h: int = 1, num_rays_x: int = 16, num_rays_y: int = 9,
                    distortion_params: Optional[Tuple[float, float, float]] = None,
                    subset_size: float = 1.0, seed: int = 42,
                    device: torch.device = "cpu"):
    """
    Parses all RealEstate10K files in a given directory and computes Plücker embeddings.

    Args:
        directory (str): Path for the directory containing txt files.
        num_rays_x (int): Number of horizontal rays. Default is 16.
        num_rays_y (int): Number of vertical rays. Default is 9.
        w (int): Width of the videos. Default is 1 (normalized).
        h (int): Height of the videos. Default is 1 (normalized).
        distortion_params (tuple or None): (d1, d2, d3) radial distortion parameters. Default to None (no distortion).
        subset_size (float): Fraction of files to process (0.1 means 10% of files). Must be greater than 0. Default is 1.0 (all files).
        seed (int): Random seed for reproducibility. Default is 42.
        device (torch.device): Device for computation (CPU/GPU). Default is "cpu".

    Returns:
        list: A list of dictionaries where each entry contains:
            - "url": (str) Video URL.
            - "K": (torch.Tensor) [3,3] Intrinsic matrix.
            - "timestamps": (np.ndarray) Frame timestamps.
            - "extrinsics": (torch.Tensor) [num_frames, 3, 4] Camera extrinsics.
            - "plucker": (torch.Tensor) [num_frames, 6, num_rays_y, num_rays_x] Plucker embeddings.
    """
    
    camera_data = []
    txt_files = [file for file in os.listdir(directory) if file.endswith(".txt")]
    
    random.seed(seed)
    np.random.seed(seed)
    num_files = max(1, int(len(txt_files) * subset_size)) # ensure at least 1 file is selected
    txt_files = random.sample(txt_files, num_files)
    
    for txt_file in tqdm(txt_files, desc="Processing RealEstate10K files", unit="file"):
        filepath = os.path.join(directory, txt_file)
        
        video_url, K, timestamps, extrinsics = _get_camera_params_for_file(filepath, w, h, device=device)
        
        K_inv = torch.linalg.inv(K)
        num_frames = extrinsics.shape[0]
        
        homogeneous_uv = _precompute_homogeneous_uv(num_rays_x, num_rays_y, distortion_params, device=device)
        num_rays = homogeneous_uv.shape[1]

        R = extrinsics[:, :, :3] # [num_frames, 3, 3]
        t = extrinsics[:, :, 3]  # [num_frames, 3]
        
        # ray direction d = R^T * K^-1 * [u, v, 1]^T
        rays = repeat(K_inv @ homogeneous_uv, "a b -> n a b", n=num_frames) # K_inv @ homogeneous_uv has shape [3, num_rays] so we get [num_frames, 3, num_rays]
        d = torch.bmm(R.transpose(1, 2), rays) # [num_frames, 3, num_rays]
        d = d.transpose(1, 2)
        d = d / torch.norm(d, dim=2, keepdim=True) 
        
        # ray origins (camera center in world coordinate)
        camera_centers = -torch.einsum('nij,nj->ni', R.transpose(1, 2), t)
        O = repeat(camera_centers, "n c -> n b c", b=num_rays) # [num_frames, num_rays, 3]
        # ray moment m = o x d
        m = torch.cross(O, d, dim=2) # [num_frames, num_rays, 3], PS: since ||d||=1, then ||m||=distance from ray to origin
        
        plucker = torch.cat([d, m], dim=2)
        plucker_embeddings = rearrange(plucker, "n (h w) c -> n c h w", h=num_rays_y, w=num_rays_x)
        
        # plucker_embeddings = torch.zeros((num_frames, 6, num_rays_y, num_rays_x), dtype=torch.float32, device=K_inv.device)
        # for i in range(num_frames):
        #     plucker_embeddings[i] = get_plucker_embedding(K_inv, extrinsics[i], num_rays_x, num_rays_y, distortion_params)

        camera_data.append({
            "url": video_url,
            "K" : K, 
            "timestamps": timestamps,
            "extrinsics": extrinsics,
            "plucker": plucker_embeddings
        })

    return camera_data


#############################
## FILE I/O ##
#############################

def save_data(data: List, filename: str):
    """
    Saves the parsed camera data to a file using pickle.

    Args:
        data (list): List of dictionaries containing camera data.
        filename (str): Name of the file to save the data.
    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Data successfully saved to {filename}")

def load_data(filename: str):
    """
    Loads the parsed camera data from a file.

    Args:
        filename (str): Name of the file to load the data from.

    Returns:
        list: List of dictionaries containing camera data.
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Data successfully loaded from {filename}")
    return data