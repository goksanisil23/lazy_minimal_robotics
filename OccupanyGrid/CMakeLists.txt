cmake_minimum_required(VERSION 3.8)
project(occupancy_grid)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -fPIC -O3")
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Carla
set(LIB_CARLA_PATH "/home/goksan/Work/carla/Examples/CppClient/libcarla-install")
set(CARLA_SIM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../Tools/carla/CarlaSim)

# Open3D
find_package(Open3D REQUIRED)
if(NOT Open3D_FOUND)
    message(FATAL_ERROR "Open3D not found, please use -DCMAKE_INSTALL_PREFIX=open3d_install_dir")
endif()

# find_package(PCL 1.3 REQUIRED)

set(LMR_UTILS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../Tools/common)

####### Build CarlaSim as library #######
add_library(carla_sim ${CARLA_SIM_PATH}/CarlaSim.cpp)
target_include_directories(carla_sim PUBLIC 
	${LIB_CARLA_PATH}/include
	${LIB_CARLA_PATH}/include/system
	# ${PCL_INCLUDE_DIRS}
	${CARLA_SIM_PATH}
	${CMAKE_CURRENT_SOURCE_DIR}/../Tools/common
)

target_link_directories(carla_sim PUBLIC
    ${LIB_CARLA_PATH}/lib
	# ${PCL_LIBRARY_DIRS}
)

target_link_libraries(carla_sim 
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
# add_definitions(${PCL_DEFINITIONS})

####### Build occupancy grid #######
add_executable(${CMAKE_PROJECT_NAME} 
	main.cpp 
	OccupancyGrid.cpp
	${LMR_UTILS_PATH}/TimeUtil.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME} 
	carla_sim
)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR}	
	${CARLA_SIM_PATH}
	${LMR_UTILS_PATH}
)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ENABLE_OGRID_VIS=1)

