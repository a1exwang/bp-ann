cmake_minimum_required(VERSION 3.5)
project(bp_ann)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES main.cpp layers/Layer.cpp layers/Layer.h layers/InputLayer.cpp layers/InputLayer.h layers/MappingInputLayer.cpp layers/MappingInputLayer.h layers/OutputLayer.cpp layers/OutputLayer.h layers/FullyConnectedLayer.cpp layers/FullyConnectedLayer.h train/SimpleTrainer.cpp train/SimpleTrainer.h utils/MathUtils.cpp utils/MathUtils.h)
link_libraries(boost_program_options)

add_executable(bp_ann ${SOURCE_FILES})
