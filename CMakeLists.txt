cmake_minimum_required(VERSION 3.2)

project(MatrixMultiply CUDA)
enable_language(CUDA)
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(include)
file(GLOB SOURCES "src/*")
add_executable(MatrixMul ${SOURCES})
target_link_libraries( MatrixMul ${OpenCV_LIBS} )