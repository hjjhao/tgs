from concurrent.futures import thread
from cv2 import threshold
import torch
import numpy as np

SMOOTH = 1e-6 # Avoid 0/0

# Pytorch Version

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    """
    平时注释掉第一行，除非输入是Batch x H x W，
    如果是UNet这种输出为Batch x 1 x H x W
    """
    outputs = outputs.squeeze(1) # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)
    
    intersection = (outputs & labels).float().sum((1, 2)) # Will be zero if Truth = 0 or Predict = 0
    union = (outputs | labels).float().sum((1, 2))        # Will be zero if both = 0

    iou = (intersection + SMOOTH) / (union + SMOOTH) 

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10 # This is equal to comparing with threshold, 将小于0.5的iou排除
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

# Numpy Version

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    return thresholded # Or thresholded.mean() if you are interested in average across the batch