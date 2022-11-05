cmake_minimum_required(VERSION 3.11)
project(sfm-example)

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

# Open3D
find_package(Open3D REQUIRED)
if(NOT Open3D_FOUND)
    message(FATAL_ERROR "Open3D not found, please use -DCMAKE_INSTALL_PREFIX=open3d_install_dir")
endif()

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

# cnpy 
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../External/cnpy ${CMAKE_CURRENT_BINARY_DIR}/cnpy)

# Matplot++
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../External/matplotplusplus ${CMAKE_CURRENT_BINARY_DIR}/matplotplusplus)

# Matplotlibcpp
# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
# find_package(Python3 COMPONENTS NumPy)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../External/Matplotlib)

# Include dirs valid for all targets
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${G2O_INCLUDE_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
message(STATUS ${G2O_LIBS})

# find_package(BLAS)
# message("------------------------------------")
# message(${BLAS_INCLUDE_DIRS})

#########################
add_executable(pose_estimator pose_estimator.cpp)
add_executable(bundle_adjuster BundleAdjuster.cpp)
add_executable(feature_matcher feature_matcher.cpp)

target_link_libraries(pose_estimator PUBLIC
    ${OpenCV_LIBS}
    ${G2O_LIBS}
    glog
    cnpy
    Open3D::Open3D
    matplot 
)
target_link_libraries(pose_estimator PUBLIC
    ${OpenCV_LIBS} 
)
target_link_libraries(bundle_adjuster PUBLIC
    ${OpenCV_LIBS}
    ${G2O_LIBS}
    glog
    Open3D::Open3D
    matplot  
)
target_link_libraries(feature_matcher PUBLIC
    ${OpenCV_LIBS}
    Open3D::Open3D
    Open3D::Open3D
    matplot      
)

target_include_directories(pose_estimator PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/../External/cnpy
${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(feature_matcher PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/../External/cnpy
${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(bundle_adjuster PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR})