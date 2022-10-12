cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(region_growing_segmentation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Ofast")

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (region_growing_segmentation region_growing_segmentation.cpp)
target_link_libraries (region_growing_segmentation ${PCL_LIBRARIES})

add_executable (project_and_color project_and_color.cpp)
target_link_libraries (project_and_color ${PCL_LIBRARIES})

add_executable (ransac_fit_plane ransac_fit_plane.cpp)
target_link_libraries (ransac_fit_plane ${PCL_LIBRARIES})

add_executable (convex_hull_2d convex_hull_2d.cpp)
target_link_libraries (convex_hull_2d ${PCL_LIBRARIES})

