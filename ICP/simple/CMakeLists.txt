cmake_minimum_required(VERSION 3.11)
project(icp_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.4)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../External/Matplotlib)
include_directories(${EIGEN3_INCLUDE_DIR})

#########################
add_executable(icp_example main_icp.cpp)
# set(raylib_VERBOSE 1)
target_link_libraries(icp_example PUBLIC 
    # raylib 
    # raylib-cpp 
    Python3::Python
    Python3::Module
    Python3::NumPy
)