#!/usr/bin/env python3

import cv2
from cv2 import cvtColor
import numpy as np
import argparse
import time
import os
import glob

FILES_DIR = '/home/goksan/Downloads/depthai-experiments/gen2-pointcloud/rgbd-pointcloud/imgs3/'
CHOSEN_DIR = FILES_DIR + "chosen/"

if not os.path.exists(CHOSEN_DIR):
    os.makedirs(CHOSEN_DIR)

files = glob.glob(CHOSEN_DIR + "/*")
for f in files:
    os.remove(f)

for file in sorted(os.listdir(FILES_DIR)):
    if 'rgb' in file:
        data = np.load(FILES_DIR + file)
        data_rgb = cvtColor(data, cv2.COLOR_RGB2BGR)
        cv2.imshow("rgb", data_rgb)
        k = cv2.waitKey(0)
        if k == ord('a'):
            continue
        if k == ord('s'):
            np.save(CHOSEN_DIR+file, data)
            file_depth = file.replace("rgb", "depth")
            depth_data = np.load(FILES_DIR + file_depth)
            np.save(CHOSEN_DIR + file_depth, depth_data)
            print("saved {} and {}".format(file, file_depth))

            # data = np.load(
            #     '/home/goksan/Downloads/depthai-experiments/gen2-pointcloud/rgbd-pointcloud/imgs/rgb_0008.npy')
            # print(data.shape)
            # print(data.dtype)

            # data = np.load(
            #     '/home/goksan/Downloads/depthai-experiments/gen2-pointcloud/rgbd-pointcloud/imgs/depth_0008.npy')
            # print(data.shape)
            # print(data.dtype)
