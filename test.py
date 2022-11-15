from PIL import Image
import cv2
import deep_learning.models.hed_unet as ts
import torch
import cv2

import numpy as np

img = cv2.imread(r"F:\YD\github_code\HED-UNet-master\HED-UNet-master\data\HenanDataset\train\images\693.tif")
model = ts.HEDUNet(3)
model.load_state_dict(torch.load(r"C:\Users\Yuanda\Desktop\100.pt", map_location=torch.device('cpu')), strict=False)
cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.waitKey()
result = model(img)
