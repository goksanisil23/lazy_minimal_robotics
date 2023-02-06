# Visual Tracking
In robotics applications, we are mainly interested in online trackers as opposed to offline trackers which can utilize "future" frames.

Online "learning" trackers can re-adjust the weights of the tracking filter online after the initialization frame with ground truth, to accomodate for the changes of the tracked object and the environment. (e.g. MOSSE, KCF, CSRT). These are mainly "correlation filters" which is re-computed as the algorithm runs.

Offline learning trackers on the other hand do not update the weights during runtime, all the learning happens offline. (e.g. Siamense networks)

## 
- MOSSE-> better at tracking fast objects than optical flow

- CSRT -> if the distance between occlusion and reappearance is big, can't recover
(recovers in in motorbike tracking behind trees but not in frisbee dataset)

- Dense Optical flow -> rescaling converges to smaller bboxes since variance is smaller there.
manages to track small motions but cant recover from occlusions or fast movements.

- Kalman Filter -> when there is considerable camera movement or abrupt object movement, deviates considerably from the object since our motion
model is simple constant velocity. 
Should only use for very short term, and later for MOT.
Still, for "easier" motions, good to see that unmeasured states vx and vy can converge to true values through measurements.
---> Show the for simple projectile case, kalman filter tracks well

> **Note**
> As the speed of DNN based object detection algorithms is increasing, long term SOT tracking is becoming less relevant since its cheaper to get detections more often. However, the techniques are still relevant for MOT case, where we need to be able to relate the objects at k to k+1. If we can track individual objects to some extend, it makes the id association in MOT case much simpler when the multi instance detection happens. 


### How to run

```bash
./tracker  --dataset_folder /home/goksan/Downloads/tracking_datasets/vot2015/road --tracker csrt
./tracker  --dataset_folder /home/goksan/Downloads/slow_traffic --tracker oflow
```

## TODO

- SORT
- DeepSORT
- Byte

- Staple
2 separate (complementary features): structure(HOG) and color distribution.
Looking at color distribution is helpful when the structure of the object deforms rapidly.
Color histogram is invariant to spatial permutations. (quantized RGB colors: 32x32x32)
A foreground and background model is iteratively learnt & updated, as 3 channel histogram weights.
Then, for a given image patch at frame k, each pixel is assigned a probability whether it belongs to the object(foreground),
based on the histogram score it gets from the background and foreground histogram models.
From this likelihood map, in order to make it look like the "response" function of the structure, integral image is used to generate a "color response".


- st-dbscan

- siamese --> in c++

### References
https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#visual-tracking
