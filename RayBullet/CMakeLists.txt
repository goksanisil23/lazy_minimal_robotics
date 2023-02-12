cmake_minimum_required(VERSION 3.10)
project(RayBullet)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall")

Find_Package(Bullet REQUIRED)
find_package(Eigen3 REQUIRED)

# Raylib
set(RAYLIB_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../External/raylib-cpp/include)
include(${RAYLIB_CPP_PATH}/CMakeLists.txt)

# LMR Tools
SET(LMR_TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Tools)

# Main
add_executable(${CMAKE_PROJECT_NAME} 
    main.cpp
    ${LMR_TOOLS_DIR}/common/TimeUtil.cpp
)

target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC
    raylib
    ${BULLET_LIBRARIES}
)

# to avoid getting warnings from raylib
target_include_directories(${CMAKE_PROJECT_NAME} SYSTEM PUBLIC 
    ${RAYLIB_CPP_PATH}
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../External/Argparse
    ${LMR_TOOLS_DIR}/common/
    ${BULLET_INCLUDE_DIR}
)

target_compile_definitions(${CMAKE_PROJECT_NAME} PUBLIC 
    ${BULLET_DEFINITIONS}
)

target_link_directories(${CMAKE_PROJECT_NAME} PUBLIC 
    ${BULLET_LIBRARY_DIRS}
)