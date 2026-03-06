cmake_minimum_required(VERSION 3.16)
project(TensorLib CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Optional: enable OpenMP for #pragma omp simd vectorisation hints ──────────
# The library compiles and runs correctly without OpenMP; the pragmas are simply
# ignored.  Uncomment the block below to enable it.
#
# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     message(STATUS "OpenMP found – SIMD pragmas will be active")
# endif()

# Shared compile options helper
set(OPT_FLAGS
    $<$<CXX_COMPILER_ID:GNU,Clang>: -O3 -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:      /O2 /arch:AVX2>
)

# ── Main demo executable ──────────────────────────────────────────────────────
add_executable(main main.cpp)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(main PRIVATE ${OPT_FLAGS})

# ── Unified test runner ───────────────────────────────────────────────────────
# run_all_tests.cpp #includes the individual test .cpp files directly,
# so it needs only a single compilation unit.
add_executable(run_all_tests tests/run_all_tests.cpp)
target_include_directories(run_all_tests PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(run_all_tests PRIVATE ${OPT_FLAGS})

# Register with CTest so you can also run "ctest" from the build directory
enable_testing()
add_test(NAME TensorLibTests COMMAND run_all_tests)

# ── Usage ─────────────────────────────────────────────────────────────────────
# mkdir build && cd build
# cmake ..
# cmake --build . --config Release
#
# Run main demo:    ./Release/main.exe        (Windows)  or  ./main  (Linux)
# Run all tests:    ./Release/run_all_tests.exe           or  ./run_all_tests
# Run via CTest:    ctest --output-on-failure
