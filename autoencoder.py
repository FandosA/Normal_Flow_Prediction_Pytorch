# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:04:33 2023

@author: afandos
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += x

        out = self.relu(out)
        
        return out
    
    
class ResidualBlockTransposed(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(ResidualBlockTransposed, self).__init__()
        
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += x

        out = self.relu(out)
        
        return out
    
    
class AutoEncoder(nn.Module):
    """
    This autoencoder creates the normal flow matrix from two frames.
    Frames are (3, 480, 640), they are concatenated so that a tensor of size
    (6, 480, 640) is obtained, and then features are extracted and
    interpolated to get the normal flow (2, 480, 640).
    """
    def __init__(self, in_channels=6, out_channels=2):

        super(AutoEncoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5,5), stride=(2,2))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.residualBlock1 = ResidualBlock(128, 128)
        self.residualConv1 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(2,2), padding=1)
        self.residualBlock2 = ResidualBlock(256, 256)
        self.residualConv2 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=(2,2), padding=1)

        # Decoder
        self.residualBlockTransposed1 = ResidualBlockTransposed(512, 512)
        self.residualConvTransposed1 = nn.ConvTranspose2d(512, 256, kernel_size=(3,3), stride=(2,2), padding=1)
        self.residualBlockTransposed2 = ResidualBlockTransposed(256, 256)
        self.residualConvTransposed2 = nn.ConvTranspose2d(256, 128, kernel_size=(3,3), stride=(2,2), padding=1)
        
        self.convTransposed1 = nn.ConvTranspose2d(128, 64, kernel_size=(5,5), stride=(2,2))
        self.bnTransposed1 = nn.BatchNorm2d(64)
        self.reluTransposed1 = nn.ReLU()
        
        self.convTransposed2 = nn.ConvTranspose2d(64, 32, kernel_size=(7,7), stride=(2,2))
        self.bnTransposed2 = nn.BatchNorm2d(32)
        self.reluTransposed2 = nn.ReLU()
        
        self.convTransposed3 = nn.ConvTranspose2d(32, out_channels, kernel_size=(4,4), stride=(1,1), padding=1)

    def forward(self, frame_ant, frame):
        
        x = torch.cat((frame_ant, frame), dim=1)
        
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        
        x = self.residualBlock1(x)
        x = self.residualConv1(x)
        x = self.residualBlock2(x)
        x = self.residualConv2(x)
        
        x = self.residualBlockTransposed1(x)
        x = self.residualConvTransposed1(x)
        x = self.residualBlockTransposed2(x)
        x = self.residualConvTransposed2(x)
        
        x = self.reluTransposed1(self.bnTransposed1(self.convTransposed1(x)))
        x = self.reluTransposed2(self.bnTransposed2(self.convTransposed2(x)))
        
        x = self.convTransposed3(x)
        
        return x