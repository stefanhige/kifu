cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(kifu)

set(CMAKE_CXX_STANDARD 17)  #necessary due to std::filesystem
add_compile_options(-Wall -Wextra -Wno-sign-compare -pedantic)

set(LIB_DIR libs)
set(TEST_DIR test)
set(EIGEN_DOWNLOAD_DIR Eigen)
set(FLANN_DOWNLOAD_DIR flann_src)
set(FLANN_BUILD_DIR flann_build)
set(FLANN_INSTALL_DIR flann)
set(GTEST_DOWNLOAD_DIR googletest-download)
set(GTEST_SOURCE_DIR googletest-src)
set(GTEST_BUILD_DIR googletest-build)
set(GTEST_INSTALL_DIR gtest)

########################################### EIGEN #######################################

#Technique similiar to a google test setup with automatic download and install
#Thanks to: https://chromium.googlesource.com/external/github.com/google/googletest/+/HEAD/googletest/README.md
# Download and unpack eigen at configure time
if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${EIGEN_DOWNLOAD_DIR}/CMakeLists.txt ) # We assume it's downloaded if CMakeLists.txt is present!
    MESSAGE("Downloading Files from Eigen Git Repo...")
    configure_file(CMakeLists.txt.Eigen ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${EIGEN_DOWNLOAD_DIR}/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_GENERATOR} .
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${EIGEN_DOWNLOAD_DIR}
    )
   #setting error_quiet here is arguably a buggy solution. It is not possible with this method otherwise however, as some git files are copied right in the end, that don't exist in the directory anymore. This is not the case for the google test setup however
   execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${EIGEN_DOWNLOAD_DIR}
        ERROR_QUIET
        )
endif()

set(Eigen3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${EIGEN_DOWNLOAD_DIR})

########################################### FLANN #######################################

if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_DOWNLOAD_DIR}/CMakeLists.txt)
    MESSAGE("Downloading files from flann git repo...")
    # this moves the file
    configure_file(CMakeLists.txt.flann ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_DOWNLOAD_DIR}/CMakeLists.txt)

    execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_GENERATOR} . WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_DOWNLOAD_DIR}
        )
    # this downloads the git repo
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_DOWNLOAD_DIR}
        )
endif()
if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_BUILD_DIR}/Makefile)
    message("patch flann")
    execute_process(COMMAND patch -p2 -i ../../flann.patch
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_DOWNLOAD_DIR}
        OUTPUT_VARIABLE output
        )
    message(${output})

    message("call cmake for flann")
    execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_DOWNLOAD_DIR}
              -D CMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_INSTALL_DIR}
              -D BUILD_DOC=0
              -D BUILD_EXAMPLES=0
              -D BUILD_TESTS=0
              -D BUILD_MATLAB_BINDINGS=0
              -D BUILD_PYTHON_BINDINGS=0
              WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_BUILD_DIR}
          )
    #make install
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_BUILD_DIR} --target install)
endif()
set(FLANN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_INSTALL_DIR}/include)
set(FLANN_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_INSTALL_DIR}/lib)

################################# UNIT TESTS ########################################
#Set up Google test installation
#Thanks to: https://chromium.googlesource.com/external/github.com/google/googletest/+/HEAD/googletest/README.md

# Download and unpack googletest at configure time
if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${GTEST_DOWNLOAD_DIR}/CMakeLists.txt)

configure_file(CMakeLists.txt.gtest ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${GTEST_DOWNLOAD_DIR}/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${GTEST_DOWNLOAD_DIR}"
RESULT_VARIABLE result
)
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${GTEST_DOWNLOAD_DIR}"
)

endif()

set(GTEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${GTEST_INSTALL_DIR}/include)
set(GTEST_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${GTEST_INSTALL_DIR}/lib)

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/googletest-src"
                 "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/googletest-build"
)

add_subdirectory(${TEST_DIR})



include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/utils)

set(HEADERS
    src/utils/Eigen.h
    src/utils/FreeImageHelper.h
    src/utils/VirtualSensor.h
    src/utils/NearestNeighbor.h
    src/utils/SimpleMesh.h
    src/utils/StopWatch.h
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
    KinectFusion.cpp
)


find_package(OpenMP)

link_directories(${FreeImage_LIBRARY_DIR})

add_executable(kifu ${HEADERS} ${SOURCES})

target_include_directories(kifu PUBLIC ${Eigen3_INCLUDE_DIR})
target_include_directories(kifu SYSTEM PUBLIC ${FLANN_INCLUDE_DIR})

target_link_libraries(kifu PUBLIC general freeimage lz4 -L${FLANN_LIB_DIR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(kifu PUBLIC OpenMP::OpenMP_CXX)
endif()
