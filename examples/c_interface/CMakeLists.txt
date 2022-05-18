
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

project(libsdp_c_interface)
add_executable(libsdp_c_interface main.cc)

# libsdp
add_subdirectory(libsdp)
target_link_libraries(libsdp_c_interface PRIVATE sdp)

