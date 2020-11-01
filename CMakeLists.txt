cmake_minimum_required(VERSION 3.6)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
set(CMAKE_BUILD_PARALLEL_LEVEL 4)



set(PROJECT_NAME kifu)
set(PROJECT_LIB kifuLib)

set(LIB_DIR libs)
set(TEST_DIR test)
set(PROJECT_LIB_DIR ProjectLibrary)
set(PROJECT_EXE_DIR ProjectExecutable)

set(EIGEN_RECIPE_DIR eigen-recipe)
set(EIGEN_SOURCE_DIR eigen)

set(FLANN_RECIPE_DIR flann-recipe)
set(FLANN_SOURCE_DIR flann-source)
set(FLANN_BUILD_DIR flann-build)
set(FLANN_INSTALL_DIR flann)

set(GTEST_RECIPE_DIR googletest-recipe)
set(GTEST_SOURCE_DIR googletest-source)

project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)  #necessary due to std::filesystem
add_compile_options(-Wall -Wextra -Wno-sign-compare -pedantic)

########################################### EIGEN #######################################

#Technique similiar to a google test setup with automatic download and install
#Thanks to: https://chromium.googlesource.com/external/github.com/google/googletest/+/HEAD/googletest/README.md
# Download and unpack eigen at configure time
if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${EIGEN_SOURCE_DIR}/CMakeLists.txt ) # We assume it's downloaded if CMakeLists.txt is present!
    MESSAGE("Downloading files from Eigen git repo...")
    configure_file(CMakeLists.txt.Eigen ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${EIGEN_RECIPE_DIR}/CMakeLists.txt)

    execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_GENERATOR} . WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${EIGEN_RECIPE_DIR})
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${EIGEN_RECIPE_DIR})
else()
    message("Eigen is already downloaded.")
endif()

set(Eigen3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${EIGEN_SOURCE_DIR})

########################################### FLANN #######################################

if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_SOURCE_DIR}/CMakeLists.txt)
    MESSAGE("Downloading files from flann git repo...")
    # this moves the file
    configure_file(CMakeLists.txt.flann ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_RECIPE_DIR}/CMakeLists.txt)

    execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_GENERATOR} . WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_RECIPE_DIR})
    # this downloads the git repo
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_RECIPE_DIR})
else()
    message("Flann is already downloaded.")
endif()

if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_BUILD_DIR}/Makefile)
    message("patch flann.")
    execute_process(COMMAND patch -p2 -i ../../flann.patch
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_SOURCE_DIR}
        OUTPUT_VARIABLE output
        )
    message(${output})

    message("call cmake for flann")
    execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_SOURCE_DIR}
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
else()
    message("Flann is already built.")
endif()
set(FLANN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_INSTALL_DIR}/include)
set(FLANN_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${FLANN_INSTALL_DIR}/lib)

################################# UNIT TESTS ########################################
#Set up Google test installation
#Thanks to: https://chromium.googlesource.com/external/github.com/google/googletest/+/HEAD/googletest/README.md

# Download and unpack googletest at configure time
if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${GTEST_RECIPE_DIR}/CMakeLists.txt)
    configure_file(CMakeLists.txt.gtest ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${GTEST_RECIPE_DIR}/CMakeLists.txt)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${GTEST_RECIPE_DIR}"
        RESULT_VARIABLE result)
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" --build . WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${GTEST_RECIPE_DIR}")
    message("patch googletest")
    execute_process(COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/googletest.patch WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${GTEST_SOURCE_DIR})
endif()

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR}/${GTEST_SOURCE_DIR}")

add_subdirectory(${TEST_DIR})

add_subdirectory(${PROJECT_LIB_DIR})

target_include_directories(${PROJECT_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_LIB_DIR})

add_subdirectory(${PROJECT_EXE_DIR})
