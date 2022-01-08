cmake_minimum_required(VERSION 3.11)
project(ekf_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
# find_package(PythonLibs 2.7)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../External/Matplotlib)
include_directories(${EIGEN3_INCLUDE_DIR})

#########################
add_executable(ekf_test ekf_test.cpp)
# set(raylib_VERBOSE 1)
target_link_libraries(ekf_test PUBLIC 
    # raylib 
    # raylib-cpp 
    Python3::Python
    Python3::Module
    Python3::NumPy
)