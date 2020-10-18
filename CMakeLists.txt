cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(kifu)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR} libs CACHE PATH "Path to lib folder")
set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/Eigen/ CACHE PATH "Path to Eigen source folder")

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

add_compile_options(-Wall -Wextra -Wno-sign-compare -pedantic)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/utils)

set(HEADERS
    src/utils/Eigen.h
    src/utils/FreeImageHelper.h
    src/utils/VirtualSensor.h
    src/utils/NearestNeighbor.h
    src/utils/SimpleMesh.h
    src/DataTypes.h
    src/SurfaceReconstructor.h
    src/SurfaceMeasurer.h
    src/PoseEstimator.h
    src/SurfacePredictor.h
    KinectFusion.h
)

set(SOURCES
    src/utils/FreeImageHelper.cpp
    src/SurfaceReconstructor.cpp
    src/SurfaceMeasurer.cpp
    src/PoseEstimator.cpp
    src/SurfacePredictor.cpp
    main.cpp
)

find_package(OpenMP)

link_directories(${FreeImage_LIBRARY_DIR})
add_executable(kifu ${HEADERS} ${SOURCES})
target_include_directories(kifu PUBLIC ${Eigen3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(kifu PUBLIC  general freeimage lz4 OpenMP::OpenMP_CXX)
else()
    target_link_libraries(kifu general freeimage lz4)
endif()

