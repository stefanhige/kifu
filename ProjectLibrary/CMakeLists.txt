

set(HEADERS
    utils/Eigen.h
    utils/FreeImageHelper.h
    utils/VirtualSensor.h
    utils/NearestNeighbor.h
    utils/SimpleMesh.h
    utils/StopWatch.h
    utils/BilateralFilter.h
    DataTypes.h
    SurfaceReconstructor.h
    SurfaceMeasurer.h
    PoseEstimator.h
    SurfacePredictor.h
    KinectFusion.h  
)

set(SOURCES
    utils/FreeImageHelper.cpp
    SurfaceReconstructor.cpp
    SurfaceMeasurer.cpp
    PoseEstimator.cpp
    SurfacePredictor.cpp
    KinectFusion.cpp
)


add_library(${PROJECT_LIB} SHARED ${HEADERS} ${SOURCES})
set_target_properties(${PROJECT_LIB} PROPERTIES LINKER_LANGUAGE CXX)

#include directories and libraries!
target_include_directories(${PROJECT_LIB} PUBLIC utils)

target_include_directories(${PROJECT_LIB} SYSTEM PUBLIC ${Eigen3_INCLUDE_DIR})
target_include_directories(${PROJECT_LIB} SYSTEM PUBLIC ${FLANN_INCLUDE_DIR})

target_link_libraries(${PROJECT_LIB} PUBLIC general freeimage lz4 -L${FLANN_LIB_DIR})


find_package(OpenMP)
link_directories(${FreeImage_LIBRARY_DIR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_LIB}  PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(OpenCV REQUIRED)
target_include_directories(${PROJECT_LIB} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_LIB}  PUBLIC ${OpenCV_LIBS})

