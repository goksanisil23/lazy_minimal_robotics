cmake_minimum_required(VERSION 3.11)
project(g2o-example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

# OpenCV
SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED )
if(OpenCV_FOUND)
   message("Found OpenCV")
   message("Includes: " ${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

# g2o
list(APPEND CMAKE_MODULE_PATH "/home/goksan/Downloads/g2o/cmake_modules")
find_package(G2O REQUIRED)
set(G2O_LIBS
${G2O_CORE_LIBRARY}
${G2O_STUFF_LIBRARY}
${G2O_CLI_LIBRARY}
${G2O_SOLVER_CHOLMOD}
${G2O_SOLVER_CSPARSE}
${G2O_SOLVER_CSPARSE_EXTENSION}
${G2O_SOLVER_DENSE}
${G2O_SOLVER_PCG}
${G2O_SOLVER_SLAM2D_LINEAR}
${G2O_SOLVER_STRUCTURE_ONLY}
${G2O_SOLVER_EIGEN}
${G2O_TYPES_DATA}
${G2O_TYPES_ICP}
${G2O_TYPES_SBA}
${G2O_TYPES_SCLAM2D}
${G2O_TYPES_SIM3}
${G2O_TYPES_SLAM2D}
${G2O_TYPES_SLAM3D}
)

# Matplot++
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../External/matplotplusplus ${CMAKE_CURRENT_BINARY_DIR}/matplotplusplus)

# Include dirs valid for all targets
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${G2O_INCLUDE_DIR})
message(STATUS ${G2O_LIBS})

#########################
# add_executable(${CMAKE_PROJECT_NAME} g2o_curve_fitting.cpp)
add_executable(${CMAKE_PROJECT_NAME} g2o_1d.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
    ${G2O_LIBS}
    matplot
    glog
)

target_link_directories(${CMAKE_PROJECT_NAME} PUBLIC
	${OpenCV_LIBRARY_DIRS}
)