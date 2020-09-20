cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(kifu)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR} libs CACHE PATH "Path to lib folder")
set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/Eigen/ CACHE PATH "Path to Eigen source folder")

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

set(HEADERS 
    Eigen.h
    FreeImageHelper.h
    VirtualSensor.h
    KinectFusion.h
    NearestNeighbor.h
)

set(SOURCES
    main.cpp
    FreeImageHelper.cpp
)

find_package( OpenCV REQUIRED)


link_directories(${FreeImage_LIBRARY_DIR})
add_executable(kifu ${HEADERS} ${SOURCES})
target_include_directories(kifu PUBLIC ${Eigen3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${Flann_INCLUDE_DIR})
target_link_libraries(kifu general freeimage lz4 ${OpenCV_LIBS})

