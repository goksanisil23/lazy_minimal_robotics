cmake_minimum_required(VERSION 3.11)
project(kalman_example)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Eigen3 REQUIRED)
# find_package(PythonLibs 2.7)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../External/Raylib/CMakeLists.txt)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../External/Matplotlib)


include_directories(${EIGEN3_INCLUDE_DIR})

#########################
add_executable(kalman_test kalman_test.cpp kalman.cpp)
# set(raylib_VERBOSE 1)
target_link_libraries(kalman_test PUBLIC 
    raylib 
    raylib-cpp 
    Python3::Python
    Python3::Module
    Python3::NumPy
)

add_executable(kalman_test_change_R kalman_test_change_R.cpp kalman.cpp)
# set(raylib_VERBOSE 1)
target_link_libraries(kalman_test_change_R PUBLIC 
    raylib 
    raylib-cpp 
    Python3::Python
    Python3::Module
    Python3::NumPy
)