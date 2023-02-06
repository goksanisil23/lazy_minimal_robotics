cmake_minimum_required(VERSION 3.11)
project(hungarian C CXX)

set(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall")

# Eigen
find_package(Eigen3 REQUIRED)

# Raylib
set(RAYLIB_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../External/raylib-cpp/include)
include(${RAYLIB_CPP_PATH}/CMakeLists.txt)

# LMR Tools
SET(LMR_TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../Tools)

#########################
add_executable(hungarian_simple
    hungarian_simple.cpp 
    ${LMR_TOOLS_DIR}/common/TimeUtil.cpp
)
target_include_directories(hungarian_simple SYSTEM PUBLIC # to avoid getting warnings from raylib
    ${RAYLIB_CPP_PATH}
    ${EIGEN3_INCLUDE_DIR}
)
target_include_directories(hungarian_simple PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LMR_TOOLS_DIR}/common
)
target_link_libraries(hungarian_simple PUBLIC
    raylib
)
#########################
add_executable(${CMAKE_PROJECT_NAME}_test hungarian_test.cpp)

target_include_directories(${CMAKE_PROJECT_NAME}_test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)