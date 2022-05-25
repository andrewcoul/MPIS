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
net = smp.Unet(encoder_name='vgg19_bn', encoder_weights='imagenet', in_channels=3, classes=2)
net.load_state_dict(torch.load('vgg19_bn.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

# Load Images
img_path = os.listdir('.\data\PREDICT\imgs')
mask_path = os.listdir('.\data\PREDICT\mask')
pred_path = '.\data\PREDICT\preds'


# Predict and Dice
# dataset = BasicDataset('.\data\PREDICT\imgs', '.\data\PREDICT\mask', 1.0, True)
# test_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)

# print("Dice Score: ", evaluate(net, test_loader, device, pretrained=True, test=True))
def dice(img1,img2):
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    dice = (2*np.sum(intersection))/(np.sum(union)+np.sum(intersection))
    return dice

dice_score = 0

for i in range(len(img_path)):
    img = Image.open('.\data\PREDICT\imgs/' + img_path[i])
    pred = predict_img(net, img, device, 1.0, 0.5)
    pred = mask_to_image(pred)
    pred.save('.\data\PREDICT\preds/' + mask_path[i])
    mask = Image.open('.\data\PREDICT\mask/' + mask_path[i])

    dice_score += dice(pred, mask)

print(dice_score/len(img_path))
    

    




# # Import prediction and convert to binary
# pred_gray = cv2.imread('0090_OUT.tif', cv2.IMREAD_GRAYSCALE)
# (thresh, pred) = cv2.threshold(pred_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# # Import mask and convert to binary
# true_gray = cv2.imread('0090_TRUE.tif', cv2.IMREAD_GRAYSCALE)
# (thresh, true) = cv2.threshold(true_gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# score = dice(true, pred)

# print(score)