cmake_minimum_required(VERSION 3.11)
project(tracking)

# set(CMAKE_CXX_COMPILER "clang++")
# set(CMAKE_CXX_COMPILER "g++")
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release)
# endif()
set(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall")
# set(CMAKE_CXX_FLAGS "-std=c++17 -g -O3 -march=native -Wall")
# set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -O3 -march=native -mtune=native -funroll-loops -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory")

# Eigen
find_package(Eigen3 REQUIRED)

# Ceres
find_package(Ceres REQUIRED)

# OpenCV
SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED )
if(OpenCV_FOUND)
   message("Found OpenCV")
   message("Includes: " ${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# LMR Tools
SET(LMR_TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../Tools)

# Include dirs valid for all targets
include_directories(${CERES_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

FUNCTION(OCV_BUILD executable_name sources)
    message("Building target: ${executable_name}")
    message("Sources: ${sources}")
    
    add_executable(${executable_name} 
        ${sources} 
        ${LMR_TOOLS_DIR}/common/TimeUtil.cpp
    )

    target_link_libraries(${executable_name} PUBLIC
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        OpenMP::OpenMP_CXX
    )

    target_include_directories(${executable_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/Staple
        ${CMAKE_CURRENT_SOURCE_DIR}/../../External/Argparse
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${LMR_TOOLS_DIR}/common/
        # ${CMAKE_CURRENT_SOURCE_DIR}/../Tools/CarlaImageHandler
    )

    target_compile_options(${executable_name} PUBLIC ${OpenMP_CXX_FLAGS})
ENDFUNCTION()

# Target specifics

SET(TARGET_NAME generic_tracker)
SET(${TARGET_NAME}_sources 
    generic_tracker.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/Staple/fhog.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/Staple/staple_tracker.cpp
    ${LMR_TOOLS_DIR}/common/TimeUtil.cpp
    )
OCV_BUILD(${TARGET_NAME} "${${TARGET_NAME}_sources}")

SET(TARGET_NAME kalman_tracker)
SET(${TARGET_NAME}_sources 
    kalman_tracker.cpp  
    ${LMR_TOOLS_DIR}/common/TimeUtil.cpp
    )
OCV_BUILD(${TARGET_NAME} "${${TARGET_NAME}_sources}")

SET(TARGET_NAME projectile)
SET(${TARGET_NAME}_sources 
    projectile.cpp
    ${LMR_TOOLS_DIR}/common/TimeUtil.cpp
    
    )
OCV_BUILD(${TARGET_NAME} "${${TARGET_NAME}_sources}")
