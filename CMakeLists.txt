cmake_minimum_required(VERSION 3.18)

project(yolo-trt VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures for JetPack 6.0 Orin builds")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS
    /usr/include
    /usr/local/include
    /usr/include/aarch64-linux-gnu
)

find_library(TENSORRT_LIB nvinfer
  PATHS
    /usr/lib
    /usr/local/lib
    /usr/lib/aarch64-linux-gnu
)

find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin
  PATHS
    /usr/lib
    /usr/local/lib
    /usr/lib/aarch64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB OR NOT TENSORRT_PLUGIN_LIB)
  message(FATAL_ERROR "TensorRT 8.6 headers and libraries were not found. Set TENSORRT_INCLUDE_DIR/TENSORRT_LIB/TENSORRT_PLUGIN_LIB if needed.")
endif()

add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
set_target_properties(TensorRT::nvinfer PROPERTIES
  IMPORTED_LOCATION "${TENSORRT_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
)

add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
  IMPORTED_LOCATION "${TENSORRT_PLUGIN_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
)

file(GLOB_RECURSE headers CONFIGURE_DEPENDS modules/*.hpp modules/*.h extra/*.hpp extra/*.h)
file(GLOB_RECURSE sources CONFIGURE_DEPENDS modules/*.cpp modules/*.cu)

add_library(detector SHARED ${sources} ${headers})
target_include_directories(detector PRIVATE extra/ modules/ ${OpenCV_INCLUDE_DIRS})
target_compile_options(detector PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wno-write-strings>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_link_libraries(detector
  PRIVATE
    TensorRT::nvinfer
    TensorRT::nvinfer_plugin
    CUDA::cudart
    ${OpenCV_LIBS}
)
set_target_properties(detector PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
  CUDA_SEPARABLE_COMPILATION ON
  PUBLIC_HEADER "${headers}"
)

add_executable(yolo-trt samples/sample_detector.cpp)
target_include_directories(yolo-trt PRIVATE modules/ extra/)
target_link_libraries(yolo-trt PRIVATE detector ${OpenCV_LIBS})

install(TARGETS detector
  PUBLIC_HEADER DESTINATION /usr/local/include/yolo-tensorrt/
)
