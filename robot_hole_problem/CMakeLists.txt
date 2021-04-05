cmake_minimum_required(VERSION 3.10)

project(robot_hole_problem)

find_package(raylib 2.0 REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} main.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME} raylib)