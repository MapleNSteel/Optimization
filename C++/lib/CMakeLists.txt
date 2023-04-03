find_package (Eigen3 3.3 REQUIRED)

set(HEADER_FILE_NLP NLP/NLP.hpp)
add_library(NLP STATIC ${HEADER_FILE_NLP})

set_target_properties(NLP PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(NLP PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(NLP PRIVATE Eigen3::Eigen)


set(HEADER_FILE_IPOPT Solver/IPOPT.hpp)
add_library(ipopt STATIC ${HEADER_FILE_IPOPT})

set_target_properties(ipopt PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(ipopt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ipopt PRIVATE Eigen3::Eigen)