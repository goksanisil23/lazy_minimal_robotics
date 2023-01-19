# Visual Tracking
In robotics applications, we are mainly interested in online trackers as opposed to offline trackers which can utilize "future" frames.

Online "learning" trackers can re-adjust the weights of the tracking filter online after the initialization frame with ground truth, to accomodate for the changes of the tracked object and the environment. (e.g. MOSSE, KCF, CSRT). These are mainly "correlation filters" which is re-computed as the algorithm runs.

Offline learning trackers on the other hand do not update the weights during runtime, all the learning happens offline. (e.g. Siamense networks)

MOSSE-> better at tracking fast objects than optical flow

CSRT -> if the distance between occlusion and reappearance is big, can't recover
(recovers in in motorbike tracking behind trees but not in frisbee dataset)

Dense Optical flow -> rescaling converges to smaller bboxes since variance is smaller there.
manages to track small motions but cant recover from occlusions or fast movements.

### How to run
./tracker  --dataset_folder /home/goksan/Downloads/tracking_datasets/vot2015/road --tracker csrt
./tracker  --dataset_folder /home/goksan/Downloads/slow_traffic --tracker oflow


## Stationary camera

## Moving camera

## Occlusion



- SORT
- DeepSORT
https://github.com/opendr-eu/opendr/tree/master/src/opendr/perception/object_tracking_2d/deep_sort/algorithm/deep_sort/sort
- Byte

- tracking features with optical flow & mosse

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



<!-- DEFINE USE CASES TO TRACK FIRST: -->

1) vehicle tracking in stationary camera
2) ball tracking in broadcast
3) stationary webcam drink tracking
4) moving camera with zooming
5) object changin shape
5) occlusion

### References
https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#visual-tracking
