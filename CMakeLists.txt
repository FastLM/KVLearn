cmake_minimum_required(VERSION 3.16)
project(KVLearn VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KVLEARN_BUILD_TESTS "Build unit tests" ON)
option(KVLEARN_BUILD_SIM "Build end-to-end simulator" ON)

add_library(kvlearn
  src/cost_model.cpp
  src/features.cpp
  src/prp.cpp
  src/cars.cpp
  src/atc.cpp
  src/pool.cpp
  src/radix_tree.cpp
  src/kvlearn.cpp
)

target_include_directories(kvlearn
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(kvlearn PRIVATE
  -Wall -Wextra -Wpedantic
  $<$<CONFIG:Release>:-O3 -DNDEBUG>
)

if(KVLEARN_BUILD_SIM)
  add_executable(kvlearn_sim apps/simulate.cpp)
  target_link_libraries(kvlearn_sim PRIVATE kvlearn)
endif()

if(KVLEARN_BUILD_TESTS)
  enable_testing()
  add_executable(kvlearn_tests tests/test_kvlearn.cpp)
  target_link_libraries(kvlearn_tests PRIVATE kvlearn)
  add_test(NAME kvlearn_unit COMMAND kvlearn_tests)
endif()

install(TARGETS kvlearn
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
