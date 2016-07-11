cmake_minimum_required(VERSION 3.5)
project(bp_ann)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES main.cpp)
link_libraries(boost_program_options)

add_executable(bp_ann ${SOURCE_FILES})
