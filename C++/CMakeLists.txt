cmake_minimum_required(VERSION 3.9)
project(SolverLib)

find_package (Eigen3 3.3 REQUIRED)

add_subdirectory(lib)

# Define the executable target
add_executable(main examples/main.cpp)
target_link_libraries(main PRIVATE Eigen3::Eigen PUBLIC ipopt)
target_compile_features(main PRIVATE cxx_std_17)