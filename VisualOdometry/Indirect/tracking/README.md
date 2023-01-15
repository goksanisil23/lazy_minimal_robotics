# Visual Odometry Via Sparse Optical Flow Corner Feature Tracking
*The main difference of this method compared to VO via feature matching is that, after calculating the keypoints in an image, instead of computing the descriptors and doing descriptor matching, keypoints are tracked from k to k+1 via optical flow.*

## Usage in Motion Estimation
Optical flow can be utilized to keep track of the image features between successive frames, as an alternative to re-running feature extraction and feature matching in each frame.


The approach in this implementation is as follows:
- Pick the best N corners in the image via Shi-Thomasi corner detector.
- Via KLT optical flow, get the coordinates of these features in the next frame.
    - To verify the tracked points, re-run optical flow in reverse order with the result of the forward optical flow and check if the result matches with the original feature point's coordinates.
    - Eliminate the keypoints that do not match.
- use PnP with RANSAC to solve for relative pose, by utilizing 2D-3D correspondences. 3D world-points of the tracked features are obtained from the depth camera.
- When the number of tracked features fall below a threshold number, re-trigger the corner detector.
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/tracking/resources/viso_oflow.gif" width=100% height=50%>