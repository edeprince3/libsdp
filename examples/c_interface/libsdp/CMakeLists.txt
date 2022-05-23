
include(FetchContent)

message(STATUS "libsdp will be built at compile time")
FetchContent_Declare(libsdp_external
    GIT_REPOSITORY https://github.com/edeprince3/libsdp.git
    GIT_TAG 713afac964875449673932123534ecd3daa704c7
)
FetchContent_MakeAvailable(libsdp_external)

set(libsdp_external_DIR ${EXTERNAL_INSTALL_PREFIX}/share/cmake/libsdp CACHE PATH "path to internally built libsdpConfig.cmake" FORCE)

