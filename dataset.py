# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:41:21 2023

@author: afandos
"""

import os
import json
import configargparse


if __name__ == "__main__":
    
    arg = configargparse.ArgumentParser()
    arg.add_argument('--dataset_path', type=str, default='dataset/train', help='Dataset path.')
    args = arg.parse_args()
    
    scenearios = os.listdir(args.dataset_path)
    
    data_autoencoder = {
        "frame_ant": [],
        "frame": [],
        "optical_flow": []
    }
            
    for i, sceneario in enumerate(scenearios):
        
        images_path = os.path.join(args.dataset_path, sceneario, "image_left")
        images_names = os.listdir(images_path)
        
        optical_flow_path = os.path.join(args.dataset_path, sceneario, "flow")
        flow_files = os.listdir(optical_flow_path)
        
        for i in range(len(images_names) - 1):
            
            data_autoencoder["frame_ant"].append(os.path.join(images_path, images_names[i]))
            data_autoencoder["frame"].append(os.path.join(images_path, images_names[i + 1]))
            data_autoencoder["optical_flow"].append(os.path.join(optical_flow_path, flow_files[2 * i]))
    
    # Save data in json file
    with open("dataset_autoencoder.json", "w") as fp:
        json.dump(data_autoencoder, fp, indent=4)
