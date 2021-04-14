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

add_library(sdp SHARED 
    src/blas_helper.cc 
    src/lbfgs_helper.cc 
    src/cg_solver.cc 
    src/sdp_solver.cc 
    src/bpsdp_solver.cc 
    src/rrsdp_solver.cc)

# 
target_include_directories(sdp PRIVATE ${LBFGS_INCLUDE})

