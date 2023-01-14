cmake_minimum_required(VERSION 3.11)
project(tracking)

# set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_COMPILER "g++")
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++14 ${SSE_FLAGS} -g -O3 -march=native")
# set(CMAKE_CXX_FLAGS "-std=c++17 -g -march=native")
set(CMAKE_CXX_FLAGS "-std=c++17 -g -O3 -march=native -Wall")
# set(CMAKE_CXX_FLAGS "-std=c++17 -g -O3 -march=native -fopenmp=libiomp5")
# set(CMAKE_CXX_FLAGS "-std=c++17 -g -O3 -march=native -fno-inline -fno-tree-vectorize")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Eigen
find_package(Eigen3 REQUIRED)

# OpenCV
SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED )
if(OpenCV_FOUND)
   message("Found OpenCV")
   message("Includes: " ${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

# OpenMP
find_package(OpenMP REQUIRED)

SET(LMR_TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Tools)

# Include dirs valid for all targets
# include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})

message("*********************")
message("${OpenCV_LIBS}")

FUNCTION(OCV_BUILD executable_name sources)
    message("Building target: ${executable_name}")
    message("Sources: ${sources}")
    
    add_executable(${executable_name} 
        ${sources} 
        # ${LMR_TOOLS_DIR}/common/TimeUtil.cpp
    )

    target_link_libraries(${executable_name} PUBLIC
        opencv_calib3d
        opencv_core
        opencv_features2d
        opencv_ximgproc
        opencv_videoio
        opencv_highgui
        # ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
        )
    
        target_include_directories(${executable_name} PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        )

    target_include_directories(${executable_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../External/Argparse
        ${LMR_TOOLS_DIR}/common/
        # ${CMAKE_CURRENT_SOURCE_DIR}/../Tools/CarlaImageHandler
        )

    target_compile_options(${executable_name} PUBLIC ${OpenMP_CXX_FLAGS})
ENDFUNCTION()

# Target specifics

# SET(TARGET_NAME lmr_tracker_lasot)
# SET(${TARGET_NAME}_sources main_lasot_lmr.cpp)
# OCV_BUILD(${TARGET_NAME} ${${TARGET_NAME}_sources})

# SET(TARGET_NAME ocv_tracker_lasot)
# SET(${TARGET_NAME}_sources main_lasot_ocv.cpp)
# OCV_BUILD(${TARGET_NAME} ${${TARGET_NAME}_sources})

# SET(TARGET_NAME ocv_legacy_tracker_lasot)
# SET(${TARGET_NAME}_sources main_lasot_ocv_legacy.cpp)
# OCV_BUILD(${TARGET_NAME} ${${TARGET_NAME}_sources})

SET(TARGET_NAME tracker_oflow)
SET(${TARGET_NAME}_sources main_oflow.cpp)
OCV_BUILD(${TARGET_NAME} ${${TARGET_NAME}_sources})
