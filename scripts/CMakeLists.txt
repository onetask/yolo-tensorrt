cmake_minimum_required(VERSION 3.18)

project(yolo-trt-consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_library(detector SHARED IMPORTED GLOBAL)
set_target_properties(detector PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/lib/libdetector.so"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/modules;${CMAKE_SOURCE_DIR}/extra"
)

add_executable(yolo-trt-app "${CMAKE_SOURCE_DIR}/samples/sample_detector.cpp")
target_link_libraries(yolo-trt-app PRIVATE detector ${OpenCV_LIBS})
