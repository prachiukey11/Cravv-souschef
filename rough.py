
#image_shape=(1471, 1465, 3)

# 472 images, 472 masks
# 60 det images  COCO annotations
# 389 cls images across 5 folders

from PIL import Image
import numpy as np, json
import cv2

# img= cv2.imread("dataset/segmentation/images/image_1734186212166.jpg")
# mask=cv2.imread("dataset/segmentation/masks/image_1734186212166.png")

# print("==========================")
# print(img.shape, mask.shape)
# print("==========================")
# print(img.dtype, mask.dtype)
# print("==========================")
# print(np.unique(img),np.unique(mask))
# print("==========================")


# coco=json.load(open("dataset/detection/_annotations.coco.json"))
# print(coco["categories"])
# print(len(coco["annotations"]))

# import os

# base_path="dataset/classification/images/"

# for d in os.listdir(base_path):
#     folder_path=os.path.join(base_path,d)
    
#     if os.path.isdir(folder_path):
#         print(d,len(os.listdir(folder_path)))
        
# #det
# img= cv2.imread("dataset/detection/images/bubbling_1739828678154_A_jpg.rf.f9d817d02deb9e25bd75adff850fa311.jpg")


# print("==========================")
# print(img.shape)
# print("==========================")
# print(img.dtype)
# print("==========================")
# print(np.unique(img))
# print("==========================")
