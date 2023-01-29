cmake_minimum_required(VERSION 3.11)
project(autotune_ekf)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall")

# Eigen
find_package(Eigen3 REQUIRED)

# Ceres
find_package(Ceres REQUIRED)

# Matplotlib
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../External/Matplotlib)
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#########################
add_executable(autotune_ekf tuner.cpp KalmanTuner.cpp KalmanErrorTerm.cpp)
target_link_libraries(autotune_ekf PUBLIC 
    Python3::Python
    Python3::Module
    Python3::NumPy
    ${CERES_LIBRARIES}
)