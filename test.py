from PIL import Image
import os
import numpy as np
from predict import predict_img, mask_to_image
import segmentation_models_pytorch as smp
import torch
from utils.dice_score import multiclass_dice_coeff
from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader
from evaluate import evaluate

# Load Model
net = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=2)
net.load_state_dict(torch.load('resnet-50.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

# Load Images
img_path = os.listdir('.\data\PREDICT\imgs')
mask_path = os.listdir('.\data\PREDICT\mask')
pred_path = '.\data\PREDICT\preds'


def dice(img1, img2):
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    dice = (2*np.sum(intersection))/(np.sum(union)+np.sum(intersection))
    return dice

def pixel(true, pred):
    true = true/255
    pred = pred.astype('bool')

    total = true.sum()
    intersection = np.logical_and(pred, true).sum()

    return intersection/total


dice_score = 0
pixel_acc = 0

for i in range(len(img_path)):
    img = Image.open('.\data\PREDICT\imgs/' + img_path[i])
    pred = predict_img(net, img, device, 1.0, 0.5)
    pred = mask_to_image(pred)
    pred.save('.\data\PREDICT\preds/' + mask_path[i])
    mask = Image.open('.\data\PREDICT\mask/' + mask_path[i])

    dice_score += dice(np.array(pred), np.array(mask))
    pixel_acc += pixel(np.array(mask), np.array(pred))

print('Dice Score: ', dice_score/len(img_path))
print('Pixex Accuracy: ', pixel_acc/len(img_path))
