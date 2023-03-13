import model.res_unet_with_depth as unet
import config
import dataset_3 as dataset
import transform_algorithm.faster_rle as rle



import torch
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader


def mask2rle(masks):
    pixels = masks.T.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[:-1:2]
    return ' '.join(str(x) for x in runs)

def make_submission(premasks):
    submission_dict = premasks
    
    submission_csv = pd.DataFrame.from_dict(submission_dict, orient='index', columns=['rle_mask'])
    submission_csv = submission_csv.reset_index().rename(columns={'index':'id'})
    submission_csv['id'] = submission_csv['id'].str.strip('.png')
    submission_csv.to_csv(config.CSV_PATH, index=False)

def predict(net=None):
    torch.cuda.empty_cache()
    net = unet.UNet_ResNet(n_channels=config.IN_CHANNELS, n_classes=config.NUM_CLASSES).to(config.DEVICE)
    net.load_state_dict(torch.load('./saved_model/res_unet_TGS_18.pth')['net'])
    _, _, test_list = dataset.get_list()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize(config.IMAGE_RESIZE),
        transforms.ToTensor()
    ])
    _, depth_csv = dataset.get_depth()
    test_loader = DataLoader(dataset.SaltDataset(test_list, depth_csv, 'test', transform), batch_size=config.BATCH_SIZE)
    print('[INFO]Predicting...')
    premasks = {}
    net.eval()
    with torch.no_grad():
        for filenames, imgs, depths in tqdm.tqdm(test_loader):
            imgs, depths = imgs.to(config.DEVICE), depths.to(config.DEVICE)
            outputs = net([imgs, depths])
            pred = torch.sigmoid(outputs).cpu().numpy()
            for i in range(len(pred)):
                # print('pred.shape', pred[i].shape)
                pr = pred[i].squeeze(0)
                pr = cv2.resize(pr, [config.IMAGE_SIZE, config.IMAGE_SIZE])
                pr = (pr > config.THRESHOLD) * 255
                pr = pr.astype(np.uint8)
                # print('pr.shape', pr.shape)
                premasks[filenames[i]] = rle.original_rLE_encode(pr)
    print('[INFO]End')
    print('[INFO]Encoding...')
    make_submission(premasks)
    print('[INFO]End')

"""
[0, 0, 0, 0],
[0, 0, 1, 1],
[0, 0, 1, 1],
[0, 0, 0, 0]

[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 1, 1, 0],
[0, 1, 1, 0]
"""


def test_mask2rle():
    import cv2 
    import matplotlib.pyplot as plt
    img = cv2.imread('./datasets/train/images/a266a2a9df.png')
    mask = cv2.imread('./datasets/train/masks/a266a2a9df.png')
    train_csv = pd.read_csv('./datasets/train.csv')
    test_mask = np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask[mask >= 127] = 1
    # mask[mask < 127] = 0
    # mask = np.transpose(mask, (2, 0, 1))
    print('mask.type: ',type(mask))
    print('mask.shape:', mask.shape)
    
    rle_mask = rle.original_rLE_encode(mask)
    print('rle_mask: ', rle_mask)    
    print('Truth rle_mask: \n', train_csv[train_csv['id'] == 'a266a2a9df'])
if __name__ == '__main__':
    predict()
    # test_mask2rle()