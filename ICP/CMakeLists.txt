cmake_minimum_required(VERSION 3.8)
project(icp_example)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC -O3")
set(LIB_CARLA_PATH "/home/goksan/Work/carla/Examples/CppClient/libcarla-install")

find_package(Open3D REQUIRED)
if(NOT Open3D_FOUND)
    message(FATAL_ERROR "Open3D not found, please use -DCMAKE_INSTALL_PREFIX=open3d_install_dir")
endif()
# find_package(PCL 1.3 REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} main.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC 
	${LIB_CARLA_PATH}/include
	${LIB_CARLA_PATH}/include/system
	${PCL_INCLUDE_DIRS}
)

target_link_directories(${CMAKE_PROJECT_NAME} PUBLIC
    ${LIB_CARLA_PATH}/lib
	${PCL_LIBRARY_DIRS}
)

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
	# ${PCL_LIBRARIES}
	Open3D::Open3D
)

add_definitions(${PCL_DEFINITIONS})