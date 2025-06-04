cmake_minimum_required(VERSION 3.23)

set(PROJECT_NAME ascent-dolfinx)
project(${PROJECT_NAME} LANGUAGES C CXX)

set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX A6000
set(CMAKE_CUDA_STANDARD 20)

find_package(DOLFINX REQUIRED)
find_package(Ascent REQUIRED
             NO_DEFAULT_PATH
             PATHS /home/sr2120/ascent/scripts/build_ascent/install/ascent-checkout)

enable_language(CUDA)

add_custom_command(
  OUTPUT poisson.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py ${SCALAR_TYPE}
  VERBATIM
  DEPENDS poisson.py
  COMMENT "Compile poisson.py using FFCx"
)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(PETSC_SCALAR_COMPLEX EQUAL 1)
  if(PETSC_REAL_DOUBLE EQUAL 1)
    set(SCALAR_TYPE "--scalar_type=complex128")
  else()
    set(SCALAR_TYPE "--scalar_type=complex64")
  endif()
else()
  if(PETSC_REAL_DOUBLE EQUAL 1)
    set(SCALAR_TYPE "--scalar_type=float64")
  else()
    set(SCALAR_TYPE "--scalar_type=float32")
  endif()
endif()

include(CheckSymbolExists)
set(CMAKE_REQUIRED_INCLUDES ${PETSC_INCLUDE_DIRS})
check_symbol_exists(PETSC_USE_COMPLEX petscsystypes.h PETSC_SCALAR_COMPLEX)
check_symbol_exists(PETSC_USE_REAL_DOUBLE petscsystypes.h PETSC_REAL_DOUBLE)


# add_executable(${PROJECT_NAME} main.cpp ${CMAKE_CURRENT_BINARY_DIR}/poisson.c)
add_executable(${PROJECT_NAME} main_cellwise.cpp ${CMAKE_CURRENT_BINARY_DIR}/poisson.c)

target_link_libraries(${PROJECT_NAME} ascent::ascent)
target_link_libraries(${PROJECT_NAME} dolfinx)


# if cuda is in the mix:
# we need to make sure CUDA_RESOLVE_DEVICE_SYMBOLS is on for our target
# (it propgates some way magically in 3.14, but not in 3.21)
if(CMAKE_CUDA_COMPILER)
	set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "89")
endif()