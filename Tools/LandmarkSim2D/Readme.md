# 2D Landmarks Simulator
Simple landmark simulation in ROS2/Rviz for basic localization testing.

## Setup & Build
```sh
# run from within this file's directory
mkdir -p ~/ros2_ws/src
ln -s $(pwd)/LandmarkSim2D/ ~/ros2_ws/src/LandmarkSim2D
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```