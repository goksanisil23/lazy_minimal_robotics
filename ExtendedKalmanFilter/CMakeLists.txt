cmake_minimum_required(VERSION 3.11)
project(ekf_example)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall")

find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)

# Matplot++
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../External/matplotplusplus ${CMAKE_CURRENT_BINARY_DIR}/matplotplusplus)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../External/Matplotlib)

include_directories(${EIGEN3_INCLUDE_DIR})

#########################
add_executable(ekf_test ekf_test.cpp)
target_link_libraries(ekf_test PUBLIC 
    Python3::Python
    Python3::Module
    Python3::NumPy
)