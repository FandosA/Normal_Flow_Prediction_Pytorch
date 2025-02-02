# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:14:59 2023

@author: andre
"""

import os
import cv2
import torch
import utils
import numpy as np
import configargparse
from autoencoder import AutoEncoder
from torchvision.transforms import transforms


def loadImageAndNormalFlow(idx, path2image, path2image_next, path2flow):
    
    frame_ant_numpy = cv2.imread(path2image)
    frame = cv2.imread(path2image_next)
    
    optical_flow = np.load(path2flow)        
    normalFlow = utils.computeNormalFlow(frame_ant_numpy, optical_flow)
    
    frame_ant = transformer(frame_ant_numpy).float().to(device)
    frame = transformer(frame).float().to(device)
    
    return torch.unsqueeze(frame_ant, dim=0), frame_ant_numpy, torch.unsqueeze(frame, dim=0), optical_flow, normalFlow


if __name__ == "__main__":
    
    # Select parameters for training
    p = configargparse.ArgumentParser()
    p.add_argument('--dataset_path', type=str, default='dataset/test', help='Dataset path')
    p.add_argument('--log_dir', type=str, default='autoencoder', help='Name of the folder to load the model')
    p.add_argument('--checkpoint', type=str, default='checkpoint_395_best.pth', help='Checkpoint path')
    p.add_argument('--GPU', type=bool, default=True, help='True to train the model in the GPU')
    opt = p.parse_args()
    
    # Select device
    if opt.GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Device assigned: GPU (' + torch.cuda.get_device_name(device) + ')\n')
    else:
        device = torch.device("cpu")
        if not torch.cuda.is_available() and opt.GPU:
            print('GPU not available, device assigned: CPU\n')
        else:
            print('Device assigned: CPU\n')
    
    # Load the model and create the model
    model = AutoEncoder().to(device)
    state_dict = torch.load(os.path.join(opt.log_dir, "checkpoints", opt.checkpoint), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    transformer = transforms.ToTensor()
    
    hsv_optical_flow = np.zeros((480, 640, 3), dtype=np.uint8)
    hsv_optical_flow[..., 1] = 255
    
    hsv_normal = np.zeros((480, 640, 3), dtype=np.uint8)
    hsv_normal[..., 1] = 255
    
    hsv_normal_predicted = np.zeros((480, 640, 3), dtype=np.uint8)
    hsv_normal_predicted[..., 1] = 255
        
    scenarios = os.listdir(opt.dataset_path)
    
    for scenario in scenarios:
        
        path2images = os.path.join(opt.dataset_path, scenario, "image_left")
        path2flows = os.path.join(opt.dataset_path, scenario, "flow")
        images_names = os.listdir(path2images)
        flow_names = os.listdir(path2flows)
        
        for i in range(len(images_names) - 1):
            
            path2image = os.path.join(path2images, images_names[i])
            path2flow = os.path.join(path2flows, flow_names[2 * i])
            path2image_next = os.path.join(path2images, images_names[i + 1])
            frame_ant_torch, frame_ant_numpy, frame_torch, optical_flow, normalFlow = loadImageAndNormalFlow(i, path2image, path2image_next, path2flow)
        
            normal_flow_predict = model(frame_ant_torch, frame_torch)
            normal_flow_predict = torch.permute(torch.squeeze(normal_flow_predict), (1, 2, 0))
            normal_flow_predict = normal_flow_predict.detach().cpu().numpy()
            
            mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
            hsv_optical_flow[..., 0] = ang * 180 / np.pi / 2
            hsv_optical_flow[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr_optical_flow = cv2.cvtColor(hsv_optical_flow, cv2.COLOR_HSV2BGR)
            
            mag, ang = cv2.cartToPolar(normalFlow[..., 0], normalFlow[..., 1])
            hsv_normal[..., 0] = ang * 180 / np.pi / 2
            hsv_normal[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr_normal = cv2.cvtColor(hsv_normal, cv2.COLOR_HSV2BGR)
            
            mag, ang = cv2.cartToPolar(normal_flow_predict[..., 0], normal_flow_predict[..., 1])
            hsv_normal_predicted[..., 0] = ang * 180 / np.pi / 2
            hsv_normal_predicted[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr_normal_predicted = cv2.cvtColor(hsv_normal_predicted, cv2.COLOR_HSV2BGR)
            
            cv2.imshow('Frame', frame_ant_numpy)
            cv2.imshow('Optical Flow', bgr_optical_flow)
            cv2.imshow('Normal Flow', bgr_normal)
            cv2.imshow('Normal Flow Predicted', bgr_normal_predicted)
            k = cv2.waitKey(20) & 0xff
            if k == 27:
                break
            
        if k == 27:
            break
            
    cv2.destroyAllWindows()
