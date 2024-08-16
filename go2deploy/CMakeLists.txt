cmake_minimum_required(VERSION 3.5)
project(go2deploy)
set(CMAKE_PREFIX_PATH "/opt/unitree_robotics;${CMAKE_PREFIX_PATH}")

message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

find_package(pybind11 REQUIRED)
find_package(unitree_sdk2 REQUIRED)

## Set compiler to use c++ 17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test src/test.cpp)
target_link_libraries(test unitree_sdk2)

pybind11_add_module(go2py src/py.cpp)
target_link_libraries(go2py PRIVATE unitree_sdk2)