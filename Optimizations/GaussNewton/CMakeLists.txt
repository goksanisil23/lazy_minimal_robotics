cmake_minimum_required(VERSION 3.11)
project(gauss-newton)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Eigen3 REQUIRED)
# find_package(PythonLibs 2.7)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)

# OpenCV
SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED )
if(OpenCV_FOUND)
   message("Found OpenCV")
   message("Includes: " ${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

# Include dirs valid for all targets
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../External/Matplotlib)
include_directories(${EIGEN3_INCLUDE_DIR})

#########################
add_executable(${CMAKE_PROJECT_NAME} gauss_newton.cpp)
# set(raylib_VERBOSE 1)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC 
    # raylib 
    # raylib-cpp 
    ${OpenCV_LIBS}
    Python3::Python
    Python3::Module
    Python3::NumPy
)

target_link_directories(${CMAKE_PROJECT_NAME} PUBLIC
	${OpenCV_LIBRARY_DIRS}
)