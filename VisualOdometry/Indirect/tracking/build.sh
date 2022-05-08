#!/bin/bash

# Pre-requisites for this application: 
# - Carla simulator 0.9.13+
# Follow https://github.com/carla-simulator/carla/tree/master/Examples/CppClient
# to generate the required carla client libraries and their dependencies
# - Open3D c++ libraries
# Follow http://www.open3d.org/docs/latest/compilation.html#compilation
# to generate libraries 

# Then, set the path of the carla client library in the CMakeLists.txt in LIB_CARLA_PATH

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH
mkdir -p build
cd build
cmake -D CMAKE_C_COMPILER=/usr/bin/gcc-7 -D CMAKE_CXX_COMPILER=/usr/bin/g++-7 -Wno-dev ..
make -j4