cmake_minimum_required(VERSION 3.8)
project(stereo_depth)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -fPIC -Ofast")

# Carla
set(LIB_CARLA_PATH "/home/goksan/Work/carla/Examples/CppClient/libcarla-install")

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


find_package(Eigen3 3.4)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)


add_executable(${CMAKE_PROJECT_NAME} main.cpp)

add_executable(test test.cpp)
target_include_directories(test PUBLIC 
	${OpenCV_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(test ${OpenCV_LIBS})
target_link_directories(test PUBLIC
	${OpenCV_LIBRARY_DIRS}
)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC 
	${LIB_CARLA_PATH}/include
	${LIB_CARLA_PATH}/include/system
	${OpenCV_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}
	${EIGEN3_INCLUDE_DIR}
)

target_link_directories(${CMAKE_PROJECT_NAME} PUBLIC
    ${LIB_CARLA_PATH}/lib
	${OpenCV_LIBRARY_DIRS}
)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../External/Matplotlib)

target_link_libraries(${CMAKE_PROJECT_NAME} 
	carla_client
	rpc
	boost_filesystem
	png
	tiff
	jpeg
	Recast
	Detour
	DetourCrowd
	Open3D::Open3D
	${OpenCV_LIBS}
    Python3::Python
    Python3::Module
    Python3::NumPy	
)
