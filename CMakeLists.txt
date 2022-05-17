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
project(libsdp)
set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.5.0
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

set(PYBIND11_CPP_STANDARD -std=c++14)

# python interface
if ( BUILD_PYTHON_INTERFACE ) 

    pybind11_add_module(libsdp SHARED 
        src/blas_helper.cc 
        src/lbfgs_helper.cc 
        src/cg_solver.cc 
        src/sdp_solver.cc 
        src/bpsdp_solver.cc 
        src/rrsdp_solver.cc
        src/sdp_helper.cc)
    
    # BLAS
    find_package(BLAS REQUIRED)
    target_link_libraries(libsdp PRIVATE ${BLAS_LIBRARIES})
    
    # liblbfgs
    set(EXTERNAL_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/external/)
    add_subdirectory(external)
    target_link_libraries(libsdp PUBLIC LBFGS::lib)

    # libsdp 
    target_include_directories(libsdp PUBLIC "${PROJECT_SOURCE_DIR}/include")
    
else()

    add_library(sdp SHARED 
        src/blas_helper.cc 
        src/lbfgs_helper.cc 
        src/cg_solver.cc 
        src/sdp_solver.cc 
        src/bpsdp_solver.cc 
        src/rrsdp_solver.cc)
    
    # libsdp headers
    target_include_directories(sdp PUBLIC "${PROJECT_SOURCE_DIR}/include")

    # BLAS
    find_package(BLAS REQUIRED)
    target_link_libraries(sdp PRIVATE ${BLAS_LIBRARIES})
    
    # liblbfgs
    set(EXTERNAL_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/external/)
    add_subdirectory(external)
    target_link_libraries(sdp PUBLIC LBFGS::lib)

    set_target_properties(sdp PROPERTIES PUBLIC_HEADER "include/sdp_solver.h;include/rrsdp_solver.h;include/bpsdp_solver.h")

    
    install(TARGETS sdp
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

endif()
