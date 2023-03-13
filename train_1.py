import config
import dataset
import model.unet as unet
from evaluate.iou import iou_pytorch

import torch
import tqdm
import time
import os
from logs.logs import logger_init
import logging

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



def train(net=None):

    logger_init(log_filename='train_log', log_level=logging.INFO)

    net = unet.UNet(n_channels=config.IN_CHANNELS, n_classes=config.NUM_CLASSES).to(config.DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    loss_func = torch.nn.BCEWithLogitsLoss()

    train_list, valid_list, test_list = dataset.get_list()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize(config.IMAGE_RESIZE),
        transforms.ToTensor()
    ])
    train_dataset = dataset.SaltDataset(train_list, 'train', transform)
    valid_dataset = dataset.SaltDataset(valid_list, 'valid', transform)
    test_dataset = dataset.SaltDataset(test_list, 'test', transform)


    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())
                
    trainSteps = len(train_dataset) / config.BATCH_SIZE
    validSteps = len(valid_dataset) / config.BATCH_SIZE

    # print('[INFO]training the network...')
    logging.info(f"Training the network...")
    train_loss_list, valid_loss_list = [], []
    train_iou_list, valid_iou_list = [], []

    logs = {}

    for epoch in range(config.EPOCHS):
        # print("[INFO]Epoch {}/{}:".format(epoch+1, config.EPOCHS))
        logging.info("Epoch {}/{}:".format(epoch+1, config.EPOCHS))
        start_time = time.time()
        epoch_train_loss = 0.0
        epoch_train_iou = 0.0
        
        net.train()
        for imgs, masks in tqdm.tqdm(train_loader):
            imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
            outputs = net(imgs)
            loss = loss_func(outputs, masks)

            iou_pred = (torch.sigmoid(outputs) > config.THRESHOLD)
            ma = (masks > 0.5)
            iou = iou_pytorch(iou_pred, ma)

            epoch_train_loss += loss
            epoch_train_iou += iou

            opt.zero_grad()
            loss.backward()
            opt.step()

        epoch_train_loss /= trainSteps
        epoch_train_iou /= trainSteps

        net.eval()
        epoch_valid_loss = 0.0
        epoch_valid_iou = 0.0

        with torch.no_grad():
            for imgs, masks in valid_loader:
                imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
                outputs = net(imgs)
                iou_pred = (torch.sigmoid(outputs) > config.THRESHOLD)
                ma = (masks > 0.5)
                epoch_valid_loss += loss_func(outputs, masks)
                epoch_valid_iou += iou_pytorch(iou_pred, ma)
                

        epoch_valid_loss /= validSteps
        epoch_valid_iou /= validSteps

        train_loss_list.append(epoch_train_loss.cpu().detach().numpy())
        valid_loss_list.append(epoch_valid_loss.cpu().detach().numpy())

        train_iou_list.append(epoch_train_iou.cpu().detach().numpy())
        valid_iou_list.append(epoch_valid_iou.cpu().detach().numpy())
        
        end_time = time.time()

        # print("[INFO]Epoch {}/{}: ---train loss: {:.6f}  ---valid loss: {:.6f} ---time: {:.2f}".format(epoch+1, config.EPOCHS, epoch_train_loss, epoch_valid_loss, end_time - start_time))
        # logs['epoch_{}'.format(epoch + 1)] = "---train: loss={:.6f}   iou={:.6f}.  ---valid: loss={:.6f}   iou={:.6f}".format(epoch_train_loss, epoch_train_iou, epoch_valid_loss, epoch_valid_iou)

        logging.info("[INFO]---time: {:.2f}".format(end_time - start_time))
        logging.info("[INFO]---train: loss={:.6f}   iou={:.6f}".format(epoch_train_loss, epoch_train_iou))
        logging.info("[INFO]---valid: loss={:.6f}   iou={:.6f}".format(epoch_valid_loss, epoch_valid_iou))

        saved_model = {'net': net.state_dict(), 'opt': opt.state_dict()}
        torch.save(saved_model, os.path.join(config.SAVED_MODEL_PATH, 'res_unet_TGS_{}.pth'.format(epoch+1)))
        torch.save(logs, os.path.join(config.SAVED_LOGS_PATH, 'simple_unet_without_aug.'))
    logging.info("[INFO]END")

    #plot
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(train_loss_list)
    plt.plot(valid_loss_list)
    plt.title('Train Loss vs Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(["Train Loss", "Valid Loss"], loc='upper right')
    plt.savefig(os.path.join(config.SAVED_PLOT_PATH, 'loss_plot_1.png'))

#plot iou list
    plt.figure()
    plt.plot(train_iou_list)
    plt.plot(valid_iou_list)
    plt.title('Train IoU vs Valid IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend(["Train IoU", "Valid IoU"], loc='upper left')
    plt.savefig(os.path.join(config.SAVED_PLOT_PATH, 'iou_plot_1.png'))
    

if __name__ == '__main__':
    train()