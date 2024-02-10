
include(FetchContent)

message(STATUS "libsdp will be built at compile time")
FetchContent_Declare(libsdp_external
    GIT_REPOSITORY https://github.com/edeprince3/libsdp.git
    GIT_TAG c4817e2f7b1784171aca64a62e956517776a8b67
)
FetchContent_MakeAvailable(libsdp_external)

set(libsdp_external_DIR ${EXTERNAL_INSTALL_PREFIX}/share/cmake/libsdp CACHE PATH "path to internally built libsdpConfig.cmake" FORCE)

