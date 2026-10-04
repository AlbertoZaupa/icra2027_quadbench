cmake_minimum_required(VERSION 3.18)
project(quadrotor_solver_benchmark LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ACADOS_INSTALL_DIR "$ENV{ACADOS_SOURCE_DIR}" CACHE PATH "Optional acados installation prefix")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(BENCH_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Python with CasADi and plotting packages")
# Runtime search paths also help locate an installation outside /usr/local.
string(REPLACE ":" ";" ACADOS_LIBRARY_HINTS "$ENV{LD_LIBRARY_PATH}")
find_library(ACADOS_LIBRARY NAMES acados HINTS "${ACADOS_INSTALL_DIR}/lib" ${ACADOS_LIBRARY_HINTS} REQUIRED)
get_filename_component(ACADOS_LIBRARY_DIR "${ACADOS_LIBRARY}" DIRECTORY)
get_filename_component(ACADOS_PREFIX "${ACADOS_LIBRARY_DIR}" DIRECTORY)
find_path(ACADOS_INCLUDE_DIR acados_c/ocp_nlp_interface.h
  HINTS "${ACADOS_INSTALL_DIR}/include" "${ACADOS_PREFIX}/include" REQUIRED)

set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GENERATED_C "${GENERATED_DIR}/quadrotor_generated.c")
set(GENERATED_H "${GENERATED_DIR}/quadrotor_generated.h")
add_custom_command(
  OUTPUT "${GENERATED_C}" "${GENERATED_H}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${GENERATED_DIR}"
  COMMAND "${BENCH_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_model.py" --output "${GENERATED_DIR}"
  DEPENDS tools/generate_model.py
  COMMENT "Generating CasADi quadrotor model and nonlinear cost functions"
  VERBATIM)

add_library(quadrotor_core STATIC src/model.cpp src/trajectory.cpp src/acados_solver.cpp "${GENERATED_C}")
target_include_directories(quadrotor_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${GENERATED_DIR}" "${ACADOS_INCLUDE_DIR}"
  "${ACADOS_INCLUDE_DIR}/blasfeo/include" "${ACADOS_INCLUDE_DIR}/hpipm/include"
  "${ACADOS_INCLUDE_DIR}/daocp/include")
target_link_libraries(quadrotor_core PUBLIC "${ACADOS_LIBRARY}" m ${CMAKE_DL_LIBS})
target_compile_options(quadrotor_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>)
# The acados enum reserves slots for disabled backends. Availability is checked
# at runtime so the same benchmark supports all four solvers after enabling them.
target_compile_definitions(quadrotor_core PUBLIC
  ACADOS_WITH_DAOCP ACADOS_WITH_DAQP ACADOS_WITH_QPOASES
  BENCH_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  BENCH_PYTHON_EXECUTABLE="${BENCH_PYTHON_EXECUTABLE}"
  BENCH_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  BENCH_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
  BENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

add_executable(quadrotor_benchmark src/benchmark.cpp)
target_link_libraries(quadrotor_benchmark PRIVATE quadrotor_core)
add_executable(quadrotor_grid_benchmark src/benchmark.cpp)
target_compile_definitions(quadrotor_grid_benchmark PRIVATE QUADROTOR_GRID_BENCHMARK)
target_link_libraries(quadrotor_grid_benchmark PRIVATE quadrotor_core)
install(TARGETS quadrotor_benchmark quadrotor_grid_benchmark RUNTIME DESTINATION bin)

option(BUILD_TESTING "Build internal tests (excluded from the default build)" OFF)
if(BUILD_TESTING)
  enable_testing()
  add_executable(quadrotor_tests tests/tests.cpp)
  target_link_libraries(quadrotor_tests PRIVATE quadrotor_core)
  add_test(NAME unit COMMAND quadrotor_tests)
  foreach(trajectory challenging daocp-worst)
    add_test(NAME smoke_${trajectory} COMMAND quadrotor_benchmark
      --solver hpipm --trajectory ${trajectory} --laps 0.04 --repetitions 2
      --no-warmup --no-plots --output "${CMAKE_CURRENT_BINARY_DIR}/smoke-${trajectory}")
  endforeach()
  add_test(NAME grid_smoke COMMAND quadrotor_grid_benchmark
    --solver hpipm --laps 0.001 --repetitions 2 --no-warmup)
  set_tests_properties(grid_smoke PROPERTIES TIMEOUT 180)
endif()
if(BUILD_TESTING)
  add_test(NAME cli_contract COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_tests.py"
    "$<TARGET_FILE:quadrotor_benchmark>" "$<TARGET_FILE:quadrotor_grid_benchmark>")
  set_tests_properties(cli_contract PROPERTIES TIMEOUT 180)
endif()
