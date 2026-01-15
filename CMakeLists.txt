cmake_minimum_required(VERSION 3.15)
project(hapc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find pybind11 via Python
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "pybind11_DIR: ${pybind11_DIR}")
find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})

# Core C++ sources
set(HAPC_CORE_SOURCES
    src/pchal_design.cpp
    src/ridge_wrappers.cpp
    src/mkernel.cpp
    src/cross_kernel.cpp
    src/pcghal_call.cpp
    src/pcghal_classi_call.cpp
    src/fast_pchal.cpp
    src/cv_fast_pchal_python.cpp
    src/single_pcghal_cpp.cpp
    src/pcghal_cv_cpp.cpp
)

# Python module
pybind11_add_module(hapc_core src/bindings.cpp ${HAPC_CORE_SOURCES})
target_link_libraries(hapc_core PRIVATE Eigen3::Eigen)
target_include_directories(hapc_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "Eigen3 found: ${Eigen3_FOUND}")
message(STATUS "pybind11 found: ${pybind11_FOUND}")
