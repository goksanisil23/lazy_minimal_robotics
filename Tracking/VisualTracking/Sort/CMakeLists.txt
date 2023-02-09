cmake_minimum_required(VERSION 3.11)
project(sort)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall")

# Eigen
find_package(Eigen3 REQUIRED)

# Raylib
set(RAYLIB_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../External/raylib-cpp/include)
include(${RAYLIB_CPP_PATH}/CMakeLists.txt)

# LMR Tools
SET(LMR_TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../Tools)

# Hungarian Optimizer
SET(HUNGARIAN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../OptimalAssignment/Hungarian)

# Sort Tracker
SET(SORT_TRACKER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SortTracker)
add_library(sort_tracker STATIC 
    ${SORT_TRACKER_DIR}/SortTracker.cpp
    ${SORT_TRACKER_DIR}/ObjTrack.cpp
    ${LMR_TOOLS_DIR}/common/TimeUtil.cpp
)
target_include_directories(sort_tracker PUBLIC 
    ${SORT_TRACKER_DIR}
    ${HUNGARIAN_DIR}
    ${LMR_TOOLS_DIR}/common/
    ${EIGEN3_INCLUDE_DIR}
)
target_include_directories(sort_tracker SYSTEM PUBLIC 
    ${RAYLIB_CPP_PATH}
)
target_link_libraries(sort_tracker PUBLIC
    raylib
)

# Main
add_executable(${CMAKE_PROJECT_NAME} 
    sort_test.cpp
    MultiRobotEnv.cpp
    ${LMR_TOOLS_DIR}/common/TimeUtil.cpp
)

target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC
    raylib
    sort_tracker
)

# to avoid getting warnings from raylib
target_include_directories(${CMAKE_PROJECT_NAME} SYSTEM PUBLIC 
    ${RAYLIB_CPP_PATH}
)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../External/Argparse
    ${LMR_TOOLS_DIR}/common/
)