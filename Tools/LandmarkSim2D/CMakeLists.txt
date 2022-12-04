cmake_minimum_required(VERSION 3.5)
project(landmark_sim_2d)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} main.cpp 
  src/VisNode.cpp 
  src/Sim.cpp 
  src/Map.cpp 
  src/Robot.cpp)

  ament_target_dependencies(${CMAKE_PROJECT_NAME} rclcpp visualization_msgs nav_msgs)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC include)

install(TARGETS
${CMAKE_PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()