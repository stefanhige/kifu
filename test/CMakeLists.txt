project(unitTests)


set(SOURCES
    tests.cpp
    LinkTest.cpp
    TsdfTest.cpp
    BilateralFilterTest.cpp
    MockTest.cpp
)

add_executable(unitTests ${SOURCES})

target_link_libraries(unitTests gtest gtest_main gmock gmock_main ${PROJECT_LIB})

