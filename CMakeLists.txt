#cmake .. -DOPENNI_INCLUDE_DIR=/home/gf/Libraries/OpenNI2/Include -DOPENNI_LIBRARY=/home/gf/Libraries/OpenNI2/Bin/x64-Release/libOpenNI2.so -DWRAPPER_INCLUDE_DIR=/home/gf/Libraries/OpenNI2_wrapper

cmake_minimum_required(VERSION 2.8)
Project (camera_calibration_surf CXX)
add_executable (camera_calibration_surf camera_calibration_surf.cpp)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

find_package(PkgConfig)
pkg_check_modules(OPENCV REQUIRED opencv)
include_directories( ${OPENCV_INCLUDE_DIRS} )
set(OPENCV_LIBRARY ${OPENCV_LDFLAGS})

find_path(OPENNI_INCLUDE_DIR NAMES OpenNI.h)
include_directories(${OPENNI_INCLUDE_DIR})
find_library(OPENNI_LIBRARY NAMES OpenNI2 libOpenNI2 )

find_path(WRAPPER_INCLUDE_DIR NAMES openni2_wrapper.h)
include_directories(${WRAPPER_INCLUDE_DIR})
find_library(WRAPPER_LIBRARY NAMES libopenni2_wrapper.so)


target_link_libraries( camera_calibration_surf ${OPENCV_LIBRARY} ${OPENNI_LIBRARY} ${WRAPPER_LIBRARY})
