cmake_minimum_required(VERSION 3.6)

set(CMAKE_C_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/compilers/bin/nvc")
set(CMAKE_CXX_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/compilers/bin/nvc++")

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(kifu)

# Eigen3 library
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR} libs CACHE PATH "Path to lib folder")
set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/Eigen/ CACHE PATH "Path to Eigen source folder")

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

add_compile_options(-Wall -Wextra -pedantic -Minfo -acc=multicore)

set(HEADERS 
    Eigen.h
    FreeImageHelper.h
    VirtualSensor.h
    KinectFusion.h
    NearestNeighbor.h
    DataTypes.h
    SimpleMesh.h
)

set(SOURCES
    main.cpp
    FreeImageHelper.cpp
)

find_package(OpenCV REQUIRED)

find_package(OpenMP)
find_package(OpenACC REQUIRED)



link_directories(${FreeImage_LIBRARY_DIR})
add_executable(kifu ${HEADERS} ${SOURCES})
target_include_directories(kifu PUBLIC ${Eigen3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${Flann_INCLUDE_DIR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(kifu PUBLIC  general freeimage lz4 ${OpenCV_LIBS} OpenMP::OpenMP_CXX ${OpenACC_CXX_FLAGS})
else()
    target_link_libraries(kifu general freeimage lz4 ${OpenCV_LIBS} ${OpenACC_CXX_FLAGS})
endif()

