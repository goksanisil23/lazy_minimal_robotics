# Terminology
- Epipolar geometry: When 2 camera views are seeing same 3D world point (this can be either from a stereo camera setup, or a single moving camera seeing the same point at 2 instances in time), triangulation of the camera centers with the 3D world point leads to certain geometric relations that are studied under epipolar geometry.

- Epipolar line: When we form a ray from C_0 to p_0, all possible points on this line projected onto camera 1 creates an epipolar line.

- Epipole: Points where the baseline intersects the two image planes.

- Epipolar plane: Plane defined by the 3D world point seen by both cameras, and the centers of 2 cameras.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/sparse/resources/epipolar_line.png" width=50% height=50%>[1]



Homography: 



[1]: https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf