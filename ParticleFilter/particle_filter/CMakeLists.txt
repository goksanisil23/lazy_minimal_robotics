cmake_minimum_required(VERSION 3.8)
project(particle_filter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(landmarksim2d_msgs REQUIRED)
find_package(LandmarkSim2dLib REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} 
  src/particle_filter_node.cpp
  src/ParticleFilter.cpp
  src/TimeUtil.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC 
  include/)

target_compile_features(${CMAKE_PROJECT_NAME} PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(${CMAKE_PROJECT_NAME} rclcpp nav_msgs landmarksim2d_msgs std_msgs visualization_msgs)
target_link_libraries(${CMAKE_PROJECT_NAME} LandmarkSim2dLib)

install(TARGETS ${CMAKE_PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
