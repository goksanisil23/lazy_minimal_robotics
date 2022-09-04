#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import time
import os
import glob

FILES_DIR = '/home/goksan/Downloads/depthai-experiments/gen2-pointcloud/rgbd-pointcloud/imgs2/'
JPEGS_DIR = FILES_DIR + "jpegs/"
DEPTHS_DIR = FILES_DIR + "depths/"

_cX = 1917.1802978515625
_cY = 1075.5216064453125
_fX = 3094.48486328125
_fY = 3094.48486328125

imageHeightOriginal = 2160
imageWidthOriginal = 3840
imageHeight = 360
imageWidth = 640
aspRatioX = imageWidthOriginal / imageHeight
aspRatioY = imageWidthOriginal / imageWidth

_cX = _cX / aspRatioX
_cY = _cY / aspRatioY
_fX = _fX / aspRatioX
_fY = _fY / aspRatioY

if not os.path.exists(JPEGS_DIR):
    os.makedirs(JPEGS_DIR)
if not os.path.exists(DEPTHS_DIR):
    os.makedirs(DEPTHS_DIR)

files = glob.glob(JPEGS_DIR + "/*")
files += glob.glob(DEPTHS_DIR + "/*")
for f in files:
    os.remove(f)


with open(FILES_DIR+"K.txt", "w+") as intrinsics_file:
    intrinsics_file.write(str(_fX) + " 0" + " " + str(_cX) + "\n")
    intrinsics_file.write("0 " + str(_fY) + " " + str(_cY) + "\n")
    intrinsics_file.write("0 0 1")

with open(JPEGS_DIR+"image_list.txt", "w+") as image_list_file:
    for file in sorted(os.listdir(FILES_DIR)):
        if 'rgb_' in file:
            data = np.load(FILES_DIR + file)
            # new_file_name = file.replace('.npy', '.png')
            # new_file_name = file.replace('.npy', '.jpg')
            new_file_name = file.replace('.npy', '.png')
            cv2.imwrite(JPEGS_DIR + new_file_name,
                        cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
        elif 'depth_' in file:
            data = np.load(FILES_DIR + file)
            new_file_name = file.replace('.npy', '.png')
            cv2.imwrite(DEPTHS_DIR + new_file_name, data)


# data = np.load(
#     '/home/goksan/Downloads/depthai-experiments/gen2-pointcloud/rgbd-pointcloud/imgs/rgb_0008.npy')
# print(data.shape)
# print(data.dtype)

# data = np.load(
#     '/home/goksan/Downloads/depthai-experiments/gen2-pointcloud/rgbd-pointcloud/imgs/depth_0008.npy')
# print(data.shape)
# print(data.dtype)
