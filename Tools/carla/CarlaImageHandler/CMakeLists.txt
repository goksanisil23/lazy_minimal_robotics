cmake_minimum_required(VERSION 3.8)
project(carla_img_handler)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -fPIC")

# Carla
set(LIB_CARLA_PATH "/home/goksan/Work/carla/Examples/CppClient/libcarla-install")

# OpenCV
SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED )
if(OpenCV_FOUND)
   message("Found OpenCV")
   message("Includes: " ${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

find_package(Eigen3 3.4)

add_executable(carla_img_saver carla_img_saver.cpp)

target_include_directories(carla_img_saver PUBLIC 
	${LIB_CARLA_PATH}/include
	${LIB_CARLA_PATH}/include/system
	${OpenCV_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}
	${EIGEN3_INCLUDE_DIR}
	${SFML_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../../common
	${CMAKE_CURRENT_SOURCE_DIR}/../../../External/jsonhpp
)

target_link_directories(carla_img_saver PUBLIC
    ${LIB_CARLA_PATH}/lib
	${OpenCV_LIBRARY_DIRS}
)

target_link_libraries(carla_img_saver
	carla_client
	rpc
	boost_filesystem
	png
	tiff
	jpeg
	Recast
	Detour
	DetourCrowd
	# Open3D::Open3D
	${OpenCV_LIBS}
	sfml-graphics
	sfml-window 
	sfml-system
)
