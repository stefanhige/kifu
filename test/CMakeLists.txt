project(unitTests)


set(SOURCES
    tests.cpp
    LinkTest.cpp
    TsdfTest.cpp
    BilateralFilterTest.cpp
)

add_executable(unitTests ${SOURCES})

target_link_libraries(unitTests gtest gtest_main ${PROJECT_LIB})

