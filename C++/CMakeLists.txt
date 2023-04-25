cmake_minimum_required(VERSION 3.9)
project(SolverLib)

find_package (Eigen3 3.3 REQUIRED)

add_subdirectory(include)

# Define the executable target
add_executable(main_lcqp_1 examples/qp-ipopt/main_lcqp_1.cpp  examples/qp-ipopt/LCQP_1.hpp)
target_link_libraries(main_lcqp_1 PRIVATE Eigen3::Eigen PUBLIC NLP Solver)
target_compile_features(main_lcqp_1 PRIVATE cxx_std_17)

add_executable(main_qcqp_1 examples/qp-ipopt/main_qcqp_1.cpp  examples/qp-ipopt/QCQP_1.hpp)
target_link_libraries(main_qcqp_1 PRIVATE Eigen3::Eigen PUBLIC NLP Solver)
target_compile_features(main_qcqp_1 PRIVATE cxx_std_17)

add_executable(main_example_1 examples/qp-ipopt/main_example_1.cpp  examples/qp-ipopt/example_1.hpp)
target_link_libraries(main_example_1 PRIVATE Eigen3::Eigen PUBLIC NLP Solver)
target_compile_features(main_example_1 PRIVATE cxx_std_17)