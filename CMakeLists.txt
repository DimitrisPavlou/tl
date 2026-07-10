cmake_minimum_required(VERSION 3.16)
project(TensorLib CXX)

# C++20 is required: the library uses concepts (tl::Numeric) to constrain
# tensor element types. Matches the README and the C++ skill guidance.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── OpenMP: makes the #pragma omp simd / parallel-for hints real ──────────────
# The library still compiles and runs correctly without OpenMP (the pragmas are
# ignored); linking it turns on SIMD vectorisation and multi-threaded loops.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - SIMD + parallel-for hints are active")
endif()

# Shared compile options helper
set(OPT_FLAGS
    $<$<CXX_COMPILER_ID:GNU,Clang>: -O3 -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:      /O2 /arch:AVX2>
)

# Applies include dir, optimisation flags, and (if available) OpenMP to a target.
function(tl_configure target)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(${target} PRIVATE ${OPT_FLAGS})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

# ── Main demo executable ──────────────────────────────────────────────────────
add_executable(main main.cpp)
tl_configure(main)

# ── Benchmark harness ─────────────────────────────────────────────────────────
add_executable(bench bench/bench.cpp)
tl_configure(bench)

# ── Unified test runner ───────────────────────────────────────────────────────
# run_all_tests.cpp #includes the individual test .cpp files directly,
# so it needs only a single compilation unit.
add_executable(run_all_tests tests/run_all_tests.cpp)
tl_configure(run_all_tests)

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
