import torch

DATA_ROOT = './datasets'
TRAIN_DATA_PATH = './datasets/train'
TEST_DATA_PATH = './datasets/test'
DEPTH_DATA_PATH = './datasets/depths.csv'

SAVED_LOGS_PATH = './logs'
SAVED_PLOT_PATH = './plot'
SAVED_MODEL_PATH = './saved_model'
CSV_PATH = './datasets/submission.csv'

SAVED_LOGS_NAME = 'res_unet_train_log'


LR: float = 0.001
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 50
EPOCHS:int = 50

IN_CHANNELS: int = 1
NUM_CLASSES: int = 1

IMAGE_SIZE: int = 101
IMAGE_RESIZE: int = 128

TEST_SPLIT:float = 0.15
RANDOM_STATE:int = 42

THRESHOLD:float = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = True if DEVICE == 'cuda' else False