project(unitTests)


set(SOURCES
    tests.cpp
)

add_executable(unitTests ${SOURCES})
target_include_directories(unitTests PUBLIC ${GTEST_INCLUDE_DIR})
target_link_libraries(unitTests freeimage -L${GTEST_LIB_DIR})
