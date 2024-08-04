cmake_minimum_required(VERSION 3.10)
project(test)
 
# 设置 C++ 编译器选项
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
 
# 指定C++编译器
set(CMAKE_CXX_COMPILER /usr/bin/g++)
 
# 添加头文件搜索路径
include_directories(${PROJECT_SOURCE_DIR}/include)
 
 
# 指定源文件目录
set(CMAKE_SOURCE_DIR src/test.cpp)
 
 
# 添加源文件
add_executable(test ${CMAKE_SOURCE_DIR})

# set output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)