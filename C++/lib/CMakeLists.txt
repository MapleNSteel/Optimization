find_package (Eigen3 3.3 REQUIRED)

set(HEADER_FILE_NLP NLP/QP.hpp)
add_library(NLP INTERFACE)

set_target_properties(NLP PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(NLP INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(NLP INTERFACE Eigen3::Eigen)


set(HEADER_FILE_IPOPT Solver/IPOPT.hpp)
add_library(Solver INTERFACE ${HEADER_FILE_IPOPT})

set_target_properties(Solver PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Solver INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Solver INTERFACE Eigen3::Eigen)