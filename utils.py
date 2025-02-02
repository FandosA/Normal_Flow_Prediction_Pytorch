# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 17:38:15 2023

@author: andre
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def computeImageGradients(image):
    """
    Computes the image gradients. If the image is RGB the size is (H,W,3),
    otherwise if the image is grayscale the size is (H,W)
    :param image: image to compute the gradients
    :param rgb_img: bool to indicate whether the image is in RGB or grayscale
    :return: image gradients, with values of type uint8 (0-255)
    """
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(sobel_x)
    
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_y = cv2.convertScaleAbs(sobel_y)
    
    return grad_x, grad_y


def computeNormalFlow(image, optical_flow):
    """
    Computes the normal flow from optical flow. If the image is RGB the size is
    (H,W,3), otherwise the image is grayscale and its size is (H,W)
    :param image: input image to compute the normal flow
    :param optical_flow: optical flow of the image. Size is (H,W,2)
    :param rgb_img: bool to indicate whether the image is in RGB or grayscale
    :return: normal flow. Size is (H,W,2)
    """
    Ix, Iy = computeImageGradients(image)
    Ix = np.float64(Ix)
    Iy = np.float64(Iy)
    
    denominator = np.sqrt(Ix ** 2 + Iy ** 2)
    
    Ix[denominator < 70] = 0
    Iy[denominator < 70] = 0
    
    scalar = (optical_flow[:, :, 0] * Ix + optical_flow[:, :, 1] * Iy) / np.where(denominator == 0.0, 1.0, denominator ** 2)
    scalar[denominator == 0.0] = 0.0
    
    normal_flow_x = scalar * Ix
    normal_flow_y = scalar * Iy
    
    normal_flow = np.zeros((image.shape[0], image.shape[1], 2))
    normal_flow[:, :, 0] = normal_flow_x
    normal_flow[:, :, 1] = normal_flow_y
    
    return normal_flow


def createModelFolder(log_dir):
    """
    Creates a folder to save the checkpoints and loss values during training
    :param log_dir: name of the folder to be created
    :return: 
    """
    os.mkdir(log_dir)
    
    checkpoints_path = os.path.join(log_dir, 'checkpoints')
    os.mkdir(checkpoints_path)

    return checkpoints_path


def saveLossValues(log_dir, train_losses, val_losses):
    """
    Saves the loss values of the training in a txt file in the folder created
    :param log_dir: name of the folder to store the txt files
    :param train_losses: array with the loss values during the training
    :param val_losses: array with the loss values during the validation
    :return: 
    """
    np.savetxt(os.path.join(log_dir, 'train_losses.txt'), train_losses)
    np.savetxt(os.path.join(log_dir, 'val_losses.txt'), val_losses)
    
    
def plotLoss(log_dir, train_losses=None, validation_losses=None):
    """
    If train_losses and validation_losses are None means that there are txt files
    with the loss values already saved in the folder, so they are loaded and the
    loss curves are shown in a plot and the image is saved too. If not, the function
    plots and saves the loss curves in a png file once the training is finished.
    :param log_dir: name of the folder to store the image or to load the loss values and plot the image
    :param train_losses: array with the loss values during the training
    :param val_losses: array with the loss values during the validation
    :return: 
    """
    if train_losses is None and validation_losses is None:
    
        files_in_dir = os.listdir(log_dir)
        
        for file in files_in_dir:
            
            if file == "train_losses.txt":
                train_losses = np.loadtxt(os.path.join(log_dir, "train_losses.txt"))
            elif file == "val_losses.txt":
                validation_losses = np.loadtxt(os.path.join(log_dir, "val_losses.txt"))
            
    epochs = np.arange(train_losses.shape[0])
    bestEpoch = np.argmin(validation_losses)
    
    plt.figure()
    plt.plot(epochs, train_losses, label="Training loss", c='b')
    plt.plot(epochs, validation_losses, label="Validation loss", c='r')
    plt.plot(bestEpoch, validation_losses[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+7, validation_losses[bestEpoch]-0.15, str(bestEpoch), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(log_dir, 'loss.png'))
    
    plt.show()
