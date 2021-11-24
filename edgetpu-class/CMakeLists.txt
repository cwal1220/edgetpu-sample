cmake_minimum_required(VERSION 2.8)
project( edgetpuclass )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Tensorflow_INCLUDE_DIRS} )
include_directories( ${CMAKE_SOURCE_DIR} )

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread -lstdc++ -lrt")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAB9 -DSTC_SERVER")

add_executable( edgetpuclass edgetpu-class.cpp )
target_link_libraries( edgetpuclass ${OpenCV_LIBS} )
target_link_libraries( edgetpuclass tensorflow-lite )
target_link_libraries( edgetpuclass ${CMAKE_SOURCE_DIR}/libedgetpu.so.1.0 )

install(TARGETS edgetpuclass DESTINATION bin)
