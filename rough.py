
#image_shape=(1471, 1465, 3)

# 472 images, 472 masks
# 60 det images  COCO annotations
# 389 cls images across 5 folders

from PIL import Image
import numpy as np, json
import cv2

img= cv2.imread("dataset/segmentation/images/image_1734186212166.jpg")
mask=cv2.imread("dataset/segmentation/masks/image_1734186212166.png")

print("==========================")
print(img.shape, mask.shape)
print("==========================")
print(img.dtype, mask.dtype)
print("==========================")
print(np.unique(img),np.unique(mask))
print("==========================")
