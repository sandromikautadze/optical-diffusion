"""
data_processing.py

This module defines the functions for extracting the information from the RealEstate10k dataset and doing the necessary processing.

Authors: Sandro Mikautadze, Elio Samaha.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
from typing import List
import random

def get_camera_params_for_file(filepath: str, w: int = 1, h: int = 1):
    """
    Parses camera parameters from a given RealEstate10K file.

    Args:
        filepath (str): Path for the txt file.
        w (int): Width of the video. Default is 1 (normalized).
        h (int): Height of the video. Default is 1 (normalized).

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
                      [0,       0,       1]], dtype=torch.float32)
    
    
    # camera extrinsics and time
    timestamps = []
    extrinsics  = []
    for line in lines[1:]: # skip the first line (video URL)
        frame = line.strip().split()
        timestamp = int(frame[0])
        timestamps.append(timestamp)
        P = torch.tensor(list(map(float, frame[7:])), dtype=torch.float32).reshape(3, 4)
        extrinsics.append(P)
    
    return url, K, np.array(timestamps), torch.stack(extrinsics)

def get_plucker_embedding(K_inv: torch.Tensor, P: torch.Tensor, num_rays_x: int = 16, num_rays_y: int = 9):
    """
    Computes Plücker ray embeddings (direction and moment) given camera intrinsics and extrinsics.

    Args:
        K_inv (torch.Tensor): 3x3 inverse intrinsic matrix.
        P (torch.Tensor): 3x4 extrinsic matrix.
        num_rays_x (int): Number of horizontal rays. Default is 16.
        num_rays_y (int): Number of vertical rays. Default is 9.

    Returns:
        tuple:
            - d (torch.Tensor): [num_rays, 3] direction vectors in world space.
            - m (torch.Tensor): [num_rays, 3] Plücker moments.
    """
    
    R, t = P[:, :3], P[:, 3]
    
    u = np.linspace(0, 1, num_rays_x, endpoint=False) + 0.5 / num_rays_x # ensure it's in the middle of the pixel
    v = np.linspace(0, 1, num_rays_y, endpoint=False) + 0.5 / num_rays_y 
    U, V = np.meshgrid(u, v)
    homogeneous_uv = np.vstack([U.flatten(), V.flatten(), np.ones_like(U.flatten())]) # [3, num_rays] where num_rays = num_rays_x * num_rays_y 
    # 0 contains the x coordinates, 1 contains the y coordinates, 2 contains the z coordinates (all 1s)
    # so homogeneous_uv[:, i] is [u_i, v_i, 1]
    
    # ray direction d = R^T * K^-1 * [u, v, 1]^T
    # K_inv = torch.linalg.inv(K)
    d = R.T @ K_inv @ torch.tensor(homogeneous_uv, dtype=torch.float32, device=K_inv.device) # [3, num rays]
    d = d / torch.norm(d, dim=0, keepdim=True) 
    d = d.T # [num_rays, 3] (each row is a direction vector)
    
    # ray origins (camera center in world coordinate)
    camera_center = - R.T @ t # [3]
    O = camera_center.expand(d.shape[0], -1) # [num_rays, 3]
    
    # ray moment m = o x d
    m = torch.cross(O, d, dim=1) # [num_rays, 3]
    # PS: since ||d||=1, then ||m||=distance from ray to origin
    
    return d, m

def parse_directory(directory: str,
                    w: int = 1, h: int = 1, num_rays_x: int = 16, num_rays_y: int = 9,
                    subset_size: float = 1.0, seed: int = 42):
    """
    Parses all RealEstate10K files in a given directory and computes Plücker embeddings.

    Args:
        directory (str): Path for the directory containing txt files.
        num_rays_x (int): Number of horizontal rays. Default is 16.
        num_rays_y (int): Number of vertical rays. Default is 9.
        w (int): Width of the videos. Default is 1 (normalized).
        h (int): Height of the videos. Default is 1 (normalized).
        subset_size (float): Fraction of files to process (0.1 means 10% of files). Must be greater than 0. Default is 1.0 (all files).
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        list: A list of dictionaries where each entry contains:
            - "url": (str) Video URL.
            - "K": (torch.Tensor) [3,3] Intrinsic matrix.
            - "timestamps": (np.ndarray) Frame timestamps.
            - "extrinsics": (torch.Tensor) [num_frames, 3, 4] Camera extrinsics.
            - "directions": (torch.Tensor) [num_frames, num_rays, 3] Ray directions.
            - "moments": (torch.Tensor) [num_frames, num_rays, 3] Plücker moments.
    """
    
    camera_data = []
    txt_files = [file for file in os.listdir(directory) if file.endswith(".txt")]
    
    random.seed(seed)
    np.random.seed(seed)
    num_files = max(1, int(len(txt_files) * subset_size)) # ensure at least 1 file is selected
    txt_files = random.sample(txt_files, num_files)
    
    for txt_file in tqdm(txt_files, desc="Processing RealEstate10K files", unit="file"):
        filepath = os.path.join(directory, txt_file)
        
        video_url, K, timestamps, extrinsics = get_camera_params_for_file(filepath, w, h)
        
        K_inv = torch.linalg.inv(K)
        num_frames = extrinsics.shape[0]
        num_rays = num_rays_x * num_rays_y
        ray_directions = torch.zeros((num_frames, num_rays, 3), dtype=torch.float32, device=K_inv.device)
        plucker_moments = torch.zeros((num_frames, num_rays, 3), dtype=torch.float32, device=K_inv.device)
        for i in range(num_frames):
            d, m = get_plucker_embedding(K_inv, extrinsics[i], num_rays_x, num_rays_y)
            ray_directions[i] = d
            plucker_moments[i] = m

        camera_data.append({
            "url": video_url,
            "K" : K, 
            "timestamps": timestamps,
            "extrinsics": extrinsics,
            "directions": ray_directions,
            "moments": plucker_moments
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