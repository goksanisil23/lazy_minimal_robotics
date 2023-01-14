set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "-std=c++17 -g -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address") ## MUST
no open mp linkage or usage

############################
set(CMAKE_CXX_COMPILER "g++")
# Optionally with
no open mp linkage or usage