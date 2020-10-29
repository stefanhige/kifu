project(unitTests)


set(SOURCES
    tests.cpp
)

add_executable(unitTests ${SOURCES})
target_link_libraries(unitTests freeimage)
