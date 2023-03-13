import train
import predict
import config
import dataset
import model.unet as unet

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def main():

    net = unet.UNet(n_channels=config.IN_CHANNELS, n_classes=config.NUM_CLASSES).to(config.DEVICE)
    train.train(net)
    predict.predict(net)

if __name__ == '__main__':
    main()