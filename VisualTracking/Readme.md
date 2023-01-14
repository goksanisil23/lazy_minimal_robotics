# Visual Tracking
In robotics applications, we are mainly interested in online trackers as opposed to offline trackers which can utilize "future" frames.

Online "learning" trackers can re-adjust the weights of the tracking filter online after the initialization frame with ground truth, to accomodate for the changes of the tracked object and the environment. (e.g. MOSSE, KCF, CSRT). These are mainly "correlation filters" which is re-computed as the algorithm runs.

Offline learning trackers on the other hand do not update the weights during runtime, all the learning happens offline. (e.g. Siamense networks)

CSRT -> a momentary occlusion loses track and never recovers

## Stationary camera

## Moving camera

## Occlusion

- GT + optical flow --> https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-tracker/
with the oflow implementation in slambook/ch8:optical_flow.cpp

- SORT
- DeepSORT
https://github.com/opendr-eu/opendr/tree/master/src/opendr/perception/object_tracking_2d/deep_sort/algorithm/deep_sort/sort
- Byte

- tracking features with optical flow & mosse

- Staple
- LDES

- st-dbscan

- siamese --> in c++

little bison: occlusion test


<!-- DEFINE USE CASES TO TRACK FIRST: -->

1) vehicle tracking in stationary camera
2) ball tracking in broadcast
3) stationary webcam drink tracking
4) moving camera with zooming
5) object changin shape
5) occlusion