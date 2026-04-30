cmake_minimum_required(VERSION 3.15)
project(hapc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Windows: Add vcpkg to CMAKE_PREFIX_PATH if it exists
if(WIN32 AND EXISTS "C:/vcpkg")
    list(APPEND CMAKE_PREFIX_PATH "C:/vcpkg/installed/x64-windows")
endif()

# Find Python.  We intentionally use the modern FindPython3 module and pass
# Python3_EXECUTABLE from setup.py so the build always targets the *same*
# interpreter that pip is using.  Without this CMake may discover a newer/
# older system Python and produce a .so tagged for the wrong ABI.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_VERSION: ${Python3_VERSION}")

# Find Eigen3 with fallback to FetchContent for Windows.
# Disabling Eigen's tests/docs avoids long Windows configuration hangs and
# matches the working CI configuration in v0.2.x wheels.
find_package(Eigen3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, downloading via FetchContent...")
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        Eigen3
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(Eigen3)
endif()

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
    src/pcghal_cv_classi_cpp.cpp
)

# Python module
pybind11_add_module(hapc_core src/bindings.cpp ${HAPC_CORE_SOURCES})
target_link_libraries(hapc_core PRIVATE Eigen3::Eigen)
target_include_directories(hapc_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "Eigen3 found: ${Eigen3_FOUND}")
message(STATUS "pybind11 found: ${pybind11_FOUND}")
