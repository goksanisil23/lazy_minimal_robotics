cmake_minimum_required(VERSION 3.11)
project(tracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen
find_package(Eigen3 REQUIRED)

# OpenCV
SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED )
if(OpenCV_FOUND)
   message("Found OpenCV")
   message("Includes: " ${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

# Matplot++
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../External/matplotplusplus ${CMAKE_CURRENT_BINARY_DIR}/matplotplusplus)

# Include dirs valid for all targets
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})

message("*********************")
message("${OpenCV_LIBS}")

FUNCTION(OCV_BUILD executable_name sources)
    message("Building target: ${executable_name}")
    message("Sources: ${sources}")
    
    add_executable(${executable_name} ${sources})

    target_link_libraries(${executable_name} PUBLIC
        ${OpenCV_LIBS}
        # matplot 
        )
    
    target_include_directories(${executable_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../External/Argparse
        # ${CMAKE_CURRENT_SOURCE_DIR}/../Tools/CarlaImageHandler
        )
ENDFUNCTION()

# Target specifics
SET(TARGET_NAME lmr_tracker)
SET(${TARGET_NAME}_sources main.cpp)
OCV_BUILD(${TARGET_NAME} ${${TARGET_NAME}_sources})

SET(TARGET_NAME lmr_tracker_lasot)
SET(${TARGET_NAME}_sources main_lasot_lmr.cpp)
OCV_BUILD(${TARGET_NAME} ${${TARGET_NAME}_sources})

SET(TARGET_NAME ocv_tracker_lasot)
SET(${TARGET_NAME}_sources main_lasot_ocv.cpp)
OCV_BUILD(${TARGET_NAME} ${${TARGET_NAME}_sources})





