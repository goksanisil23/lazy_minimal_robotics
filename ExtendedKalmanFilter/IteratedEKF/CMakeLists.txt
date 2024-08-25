cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(iterated_ekf)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++17 -g -O2")

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../../Paths.cmake)
###############################################################

add_executable(iterated_ekf main.cpp)

target_include_directories(iterated_ekf SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${EIGEN_DIR}
    # ${CERES_INCLUDE_DIRS}
)

target_link_libraries(iterated_ekf 
    ${CERES_LIBRARIES}
    raylib 
    pthread
    dl
    rt
)
target_link_directories(iterated_ekf PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(iterated_ekf PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
