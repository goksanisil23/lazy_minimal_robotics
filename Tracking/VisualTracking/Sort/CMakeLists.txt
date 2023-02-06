cmake_minimum_required(VERSION 3.11)
project(sort)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall")

# Eigen
find_package(Eigen3 REQUIRED)

# OpenCV
SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED )
if(OpenCV_FOUND)
   message("Found OpenCV")
   message("Includes: " ${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

# Raylib
set(RAYLIB_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../External/raylib-cpp/include)
include(${RAYLIB_CPP_PATH}/CMakeLists.txt)

# LMR Tools
SET(LMR_TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../Tools)

# Hungarian Optimizer
SET(HUNGARIAN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../OptimalAssignment/Hungarian)

# Include dirs valid for all targets
include_directories(${CERES_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

FUNCTION(OCV_BUILD executable_name sources)
    message("Building target: ${executable_name}")
    message("Sources: ${sources}")
    
    add_executable(${executable_name} 
        ${sources} 
        ${LMR_TOOLS_DIR}/common/TimeUtil.cpp
    )

    target_link_libraries(${executable_name} PUBLIC
        ${OpenCV_LIBS}
        raylib
    )

    # to avoid getting warnings from raylib
    target_include_directories(${executable_name} SYSTEM PUBLIC 
    ${RAYLIB_CPP_PATH}
    ${EIGEN3_INCLUDE_DIR}
    )

    target_include_directories(${executable_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../External/Argparse
        ${OpenCV_INCLUDE_DIRS}
        ${HUNGARIAN_DIR}
        ${LMR_TOOLS_DIR}/common/
    )

    target_compile_options(${executable_name} PUBLIC ${OpenMP_CXX_FLAGS})
ENDFUNCTION()

# Target specifics

SET(TARGET_NAME sort)
SET(${TARGET_NAME}_sources 
    sort_test.cpp 
    ${LMR_TOOLS_DIR}/common/TimeUtil.cpp
    MultiRobotEnv.cpp
    )
OCV_BUILD(${TARGET_NAME} "${${TARGET_NAME}_sources}")