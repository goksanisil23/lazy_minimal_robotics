cmake_minimum_required(VERSION 3.11)
project(icp_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Ofast -fopenmp -march=native")

find_package(Eigen3 3.4)
find_package(PCL 1.8 REQUIRED) # Only for visualization
find_package(OpenMP) # for MRPT

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../External/Argparse)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../External/MRPT)

#########################
add_executable(icp_example main_icp.cpp)
add_definitions(${PCL_DEFINITIONS})
target_include_directories(icp_example PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(icp_example PUBLIC 
    ${PCL_LIBRARIES}
    OpenMP::OpenMP_CXX
)