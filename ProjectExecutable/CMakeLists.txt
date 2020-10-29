add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)


target_link_libraries(${PROJECT_NAME} 
${PROJECT_LIB}
)

