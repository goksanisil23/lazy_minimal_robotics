cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(graph_slam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++17 -g -O2")

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)

################## Graph Slam ################## 

find_package(Ceres REQUIRED)

set(GRAPH_SLAM_SRCS 
    graph_slam_sim.cpp
    Agent.cpp
    Visualizer.cpp
    Odometry.cpp
    GraphSlam.cpp
)

add_executable(graph_slam ${GRAPH_SLAM_SRCS})

target_include_directories(graph_slam SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${EIGEN_DIR}
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(graph_slam 
    ${CERES_LIBRARIES}
    raylib 
    pthread
    dl
    rt)
target_link_directories(graph_slam PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(graph_slam PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
