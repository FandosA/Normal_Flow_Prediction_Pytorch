# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:46:12 2022

@author: Andres Fandos

Script to load the dataset and train the neural nentwork
"""

import os
import cv2
import json
import utils
import numpy as np
import configargparse
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from autoencoder import AutoEncoder


class Dataset(Dataset):
    
    def __init__(self, dataset_file_path, device):
        
        self.transformer = transforms.ToTensor()
        self.device = device
        
        with open(dataset_file_path, "r") as fp:
            self.data = json.load(fp)
        
    def __len__(self):
        
        return len(self.data["frame_ant"])
    
    
    def __getitem__(self, index):
        
        # Load images
        frame_ant = cv2.imread(self.data["frame_ant"][index])
        frame = cv2.imread(self.data["frame"][index])
        
        # Load optical flow and compute grouned truth normal flow
        optical_flow = np.load(self.data["optical_flow"][index])
        normalFlow = utils.computeNormalFlow(frame_ant, optical_flow)
        
        # Convert arrays from numpy to torch tensors and send them to de device assigned
        frame_ant = self.transformer(frame_ant).float().to(device)
        frame = self.transformer(frame).float().to(device)
        normalFlow = self.transformer(normalFlow).float().to(device)
        
        return frame_ant, frame, normalFlow
    


def train():
    
    min_val_loss = np.inf
    bestEpoch = 0

    train_losses = []
    val_losses = []
    
    
    print('--------------------------------------------------------------')
    
    # Loop along epochs to do the training
    for i in range(args.epochs + 1):
        
        print(f'EPOCH {i}')
        
        # Training loop
        train_loss = 0.0
        model.train()
        iteration = 1
        
        print('\nTRAINING')
        
        for frame_ant, frame, normal_flow in train_loader:
            
            print('\rEpoch[' + str(i) + '/' + str(args.epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(train_loader)), end='')
            iteration += 1
            
            frame_ant, frame, normal_flow = frame_ant.to(device), frame.to(device), normal_flow.to(device)
            
            optimiser.zero_grad()
            
            normal_flow_predict = model(frame_ant, frame)
            loss = loss_fn(normal_flow_predict, normal_flow)
            
            loss.backward()
            optimiser.step()
            
            train_loss += loss.item()
        
        
        # Validation loop
        val_loss = 0.0
        model.eval()
        iteration = 1

        print('')
        print('\nVALIDATION')
        
        for frame_ant, frame, normal_flow in validate_loader:
            
            print('\rEpoch[' + str(i) + '/' + str(args.epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(validate_loader)), end='')
            iteration += 1
            
            frame_ant, frame, normal_flow = frame_ant.to(device), frame.to(device), normal_flow.to(device)
            
            normal_flow_predict = model(frame_ant, frame)
            loss = loss_fn(normal_flow_predict, normal_flow)
            
            val_loss += loss.item()
        

        # Save loss and accuracy values
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(validate_loader))
        
        print('\n')
        print(f'- Train loss: {train_loss / len(train_loader):.3f}')
        print(f'- Validation loss: {val_loss / len(validate_loader):.3f}')
        
            
        # Save the model every 10 epochs
        if i % 20 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + ".pth"))
            
        # Save the best model when loss decreases respect to the previous best loss
        if (val_loss / len(validate_loader)) < min_val_loss:
            
            # If first epoch, save model as best, otherwise, replace the previous best model with the current one
            if i == 0:
                torch.save(model.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_best.pth"))
            else:
                os.remove(os.path.join(checkpoints_path, "checkpoint_" + str(bestEpoch) + "_best.pth"))
                torch.save(model.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_best.pth"))
            
            print(f'\nValidation loss decreased: {min_val_loss:.3f} ---> {val_loss / len(validate_loader):.3f}\nModel saved')
                
            # Update parameters with the new best model
            min_val_loss = val_loss / len(validate_loader)
            bestEpoch = i
            
        saveLossValues(args.log_dir, np.array(train_losses), np.array(val_losses))
            
        print("--------------------------------------------------------------")
    
    # Plot loss and accuracy curves
    plotLoss(args.log_dir, np.array(train_losses), np.array(val_losses))



if __name__ == "__main__":
    
    # Select parameters for training
    p = configargparse.ArgumentParser()
    p.add_argument('--dataset_file_path', type=str, default='dataset_autoencoder.json', help='Path to the dataset file.')
    p.add_argument('--train_split', type=float, default=0.9, help='Percentage of the dataset to be used for training.')
    p.add_argument('--log_dir', type=str, default='polla', help='Name of the folder to save the model.')
    p.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    p.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    p.add_argument('--epochs', type=int, default=400, help='Number of epochs.')
    p.add_argument('--GPU', type=bool, default=True, help='True to train the model in the GPU')
    args = p.parse_args()
    
    assert not (os.path.isdir(args.log_dir)), 'The folder log_dir already exists, remove it or change it'
    assert (args.train_split < 1.0), 'The percentage of the dataset to be used for training must be less than 1'
    
    # Select device
    if args.GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Device assigned: GPU (' + torch.cuda.get_device_name(device) + ')\n')
    else:
        device = torch.device("cpu")
        if not torch.cuda.is_available() and args.GPU:
            print('GPU not available, device assigned: CPU\n')
        else:
            print('Device assigned: CPU\n')
            
    # Load datasets and create dataloaders
    dataset = Dataset(args.dataset_file_path, device)
    
    total_len_dataset = len(dataset)
    len_training = int(total_len_dataset * args.train_split)
    len_validation = total_len_dataset - len_training
    
    train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [len_training,  len_validation])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=args.batch_size, shuffle=True)
    
    print('Normal flows used to train: ' + str(len(train_dataset)) + '/' + str(len(dataset)))
    print('Normal flows used to validate: ' + str(len(validate_dataset)) + '/' + str(len(dataset)) + '\n')
    
    model = AutoEncoder().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    saveLossValues = utils.saveLossValues
    plotLoss = utils.plotLoss
    checkpoints_path = utils.createModelFolder(args.log_dir)
    
    train()
