#
# @BEGIN LICENSE
#
# libsdp: a library of semidefinite programming solvers
#
# Copyright (c) 2021 by its authors (LICENSE).
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.
#
# @END LICENSE
#

cmake_minimum_required(VERSION 3.11)
project(libsdp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3") 
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3") 

find_package(OpenMP)
if (OPENMP_FOUND)
    add_compile_definitions(_OPENMP)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# python interface
if ( BUILD_PYTHON_INTERFACE ) 


    # find python and set variables for:
    #     the interpreter, include dirs, libraries, and version
    set(Python3_FIND_STRATEGY LOCATION)
    find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
    set(PYTHON_EXECUTABLE   ${Python3_EXECUTABLE})
    set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    set(PYTHON_LIBRARIES    ${Python3_LIBRARIES})
    set(PYTHON_VERSION      ${Python3_VERSION})
    
    # find pybind11 and set variables for:
    set(PYBIND11_PYTHON_VERSION 3)
    set(PYBIND11_FINDPYTHON ON)
    
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.11.1
    )
    
    FetchContent_GetProperties(pybind11)
    if(NOT pybind11_POPULATED)
        FetchContent_Populate(pybind11)
        add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    endif()
    
    set(PYBIND11_CPP_STANDARD -std=c++17)

    pybind11_add_module(_libsdp SHARED 
        libsdp/src/blas_helper.cc 
        libsdp/src/lbfgs_helper.cc 
        libsdp/src/cg_solver.cc 
        libsdp/src/sdp_solver.cc 
        libsdp/src/bpsdp_solver.cc 
        libsdp/src/rrsdp_solver.cc
        libsdp/src/sdp_helper.cc)
    
    # BLAS

    # Try to find MKL via CMake config package
    set(USE_MKL OFF)
    find_package(MKL CONFIG)
    
    if (MKL_FOUND)
        message(STATUS "MKL found via CMake config.")
        set(USE_MKL ON)
        target_link_libraries(_libsdp PRIVATE MKL::MKL)
    
    elseif(DEFINED ENV{MKLROOT})
        # Manually link to MKL if MKLROOT is set
        message(STATUS "MKLROOT found, using manual MKL linking.")
        set(USE_MKL ON)
    
        set(MKLROOT $ENV{MKLROOT})
        set(MKL_LIB_DIR "${MKLROOT}/lib/intel64")
    
        target_link_libraries(_libsdp PRIVATE
            ${MKL_LIB_DIR}/libmkl_intel_lp64.a
            ${MKL_LIB_DIR}/libmkl_core.a
            ${MKL_LIB_DIR}/libmkl_gnu_thread.a
            -fopenmp -lpthread -lm -ldl)
    
    else()
        message(STATUS "MKL not found. Falling back to generic BLAS.")
        find_package(BLAS REQUIRED)
        target_link_libraries(_libsdp PRIVATE ${BLAS_LIBRARIES})
    endif()

    # liblbfgs
    set(EXTERNAL_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/libsdp/external/)
    add_subdirectory(libsdp/external)
    target_link_libraries(_libsdp PUBLIC LBFGS::lib)

    # libsdp 
    target_include_directories(_libsdp PUBLIC "${PROJECT_SOURCE_DIR}/libsdp/include")
    
else()

    add_library(sdp SHARED 
        libsdp/include/sdp_solver.h
        libsdp/include/rrsdp_solver.h 
        libsdp/include/bpsdp_solver.h 
        libsdp/src/blas_helper.cc 
        libsdp/src/lbfgs_helper.cc 
        libsdp/src/cg_solver.cc 
        libsdp/src/sdp_solver.cc 
        libsdp/src/bpsdp_solver.cc 
        libsdp/src/rrsdp_solver.cc)
    
    # libsdp headers
    target_include_directories(sdp PUBLIC "${PROJECT_SOURCE_DIR}/libsdp/include")

    # BLAS
    find_package(BLAS REQUIRED)
    target_link_libraries(sdp PRIVATE ${BLAS_LIBRARIES})
    
    # liblbfgs
    set(EXTERNAL_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/libsdp/external/)
    add_subdirectory(libsdp/external)
    target_link_libraries(sdp PUBLIC LBFGS::lib)

    set_target_properties(sdp PROPERTIES PUBLIC_HEADER "libsdp/include/sdp_solver.h;libsdp/include/rrsdp_solver.h;libsdp/include/bpsdp_solver.h")
    
    install(TARGETS sdp
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

endif()
