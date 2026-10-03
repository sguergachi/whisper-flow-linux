cmake_minimum_required(VERSION 3.22)
project(whisper_flow_worker LANGUAGES C CXX)

# Compile the engine into our worker: no CUDA toolkit, CUDA DLL download,
# HTTP library, Python FFI structures, or MSVC installer on the user's PC.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
include(FetchContent)
find_package(SPIRV-Headers CONFIG REQUIRED)
FetchContent_Declare(whisper
    GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
    GIT_TAG f049fff95a089aa9969deb009cdd4892b3e74916)
FetchContent_MakeAvailable(whisper)
target_link_libraries(ggml-vulkan PRIVATE SPIRV-Headers::SPIRV-Headers)
add_executable(whisper-flow-worker worker.cpp)
target_compile_features(whisper-flow-worker PRIVATE cxx_std_17)
target_link_libraries(whisper-flow-worker PRIVATE whisper)
if(MINGW)
    target_link_options(whisper-flow-worker PRIVATE -municode -static-libgcc -static-libstdc++)
endif()
