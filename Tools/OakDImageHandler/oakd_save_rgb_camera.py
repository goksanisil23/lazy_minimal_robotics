#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import time


# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

# Properties
# camRgb.setPreviewSize(960, 540)
# camRgb.setInterleaved(False)
# camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setPreviewSize(960, 540)

# Linking
camRgb.preview.link(xoutRgb.input)
# camRgb.out.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    print('Connected cameras: ', device.getConnectedCameras())
    # Print out usb speed
    print('Usb speed: ', device.getUsbSpeed().name)

    dirName = "imgs"
    Path(dirName).mkdir(parents=True, exist_ok=True)

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

        # Retrieve 'bgr' (opencv format) frame
        cv2.imshow("rgb", inRgb.getCvFrame())

        if cv2.waitKey(10) == ord('s'):
            print("saving {}".format(time.time()))
            cv2.imwrite(f"{dirName}/{int(time.time() * 1000)}.png",
                        inRgb.getFrame())
