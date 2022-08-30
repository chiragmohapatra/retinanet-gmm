#Update sys path to include the pytorch RetinaNet modules
import warnings
import os
import sys

warnings.filterwarnings('ignore')
# sys.path.append("./pytorch_retinanet/")

import pandas as pd
from PIL import Image
import cv2
import numpy as np

from utils.pascal import convert_annotations_to_df

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from model import RetinaNetModel

# seed so that results are reproducible
pl.seed_everything(123)

pd.set_option("display.max_colwidth", None)
np.random.seed(123)

#Paths where to save the generated dataframes
TRAIN_CSV = "/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold_csv/0/train_data.csv"
VALID_CSV = "/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold_csv/0/valid_data.csv"
TEST_CSV  = "/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/Dataset/voc_kfold_csv/0/test_data.csv"

#load csv
train_df = pd.read_csv(TRAIN_CSV)
valid_df = pd.read_csv(VALID_CSV)
test_df = pd.read_csv(TEST_CSV)

from utils.pascal import generate_pascal_category_names

LABEL_MAP = generate_pascal_category_names(train_df)
print(LABEL_MAP)

NUM_TRAIN_EPOCHS = 1000


from omegaconf import OmegaConf

#load in the hparams.ymal file using Omegaconf
hparams = OmegaConf.load("hparams.yaml")

# ========================================================================= #
# MODIFICATION OF THE CONFIG FILE TO FIX PATHS AND DATSET-ARGUEMENTS :
# ========================================================================= #
hparams.dataset.kind        = "csv"
hparams.dataset.trn_paths   = TRAIN_CSV
hparams.dataset.valid_paths = VALID_CSV
hparams.dataset.test_paths  = TEST_CSV

hparams.dataloader.train_bs = 2
hparams.dataloader.valid_bs = 16
hparams.dataloader.test_bs  = 16

hparams.model.num_classes   = len(LABEL_MAP) - 1
hparams.model.backbone_kind = "resnet18"
hparams.model.min_size      = 800
hparams.model.max_size      = 1333
hparams.model.pretrained    = True #loads in imagenet-backbone weights

#transforms for the train_dataset
hparams.transforms  =  [
    {"class_name": "albumentations.HorizontalFlip", "params": {"p": 0.5} },
    # {"class_name": "albumentations.ShiftScaleRotate", "params": {"p": 0.5} },
    {"class_name": "albumentations.RandomBrightnessContrast", "params": {"p": 0.5} },
]

#optimizer
hparams.optimizer = {
    "class_name": "torch.optim.SGD",
    "params"    : {"lr": 0.001, "weight_decay": 0.0005, "momentum":0.9},
    }

#scheduler
hparams.scheduler = {
    "class_name" : "torch.optim.lr_scheduler.CosineAnnealingLR",
    "params"     : {"T_max": NUM_TRAIN_EPOCHS},
    "monitor"    : None,
    "interval"   : "epoch",
    "frequency"  : 1
    }

print(OmegaConf.to_yaml(hparams))

lr_logger  = LearningRateMonitor(logging_interval="step")

#instantiate LightningTrainer
trainer    = Trainer(precision=16, gpus=1, callbacks=[lr_logger], max_epochs=NUM_TRAIN_EPOCHS)
#define model
litModel = RetinaNetModel(conf=hparams)
# start training
trainer.fit(litModel)
# test model
trainer.test(litModel)

import torch
PATH = '/home/aayush/Aayush/Projects/Celiac_Disease/Detection/Baselines/RetinaNet/pytorch_retinanet/weights/Celiac_resnet18_ep1000.pth'
torch.save(litModel.net.state_dict(), PATH)

import logging
logger = logging.getLogger("lightning")
