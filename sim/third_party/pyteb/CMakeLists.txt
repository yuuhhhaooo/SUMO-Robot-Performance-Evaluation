cmake_minimum_required(VERSION 3.20)
project(pyteb CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED TEB_PREFIX)
  set(TEB_PREFIX "C:/Users/Mark/tebenv/Library")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pyteb src/pyteb.cpp)

target_include_directories(pyteb PRIVATE
  ${TEB_PREFIX}/include
  ${TEB_PREFIX}/include/eigen3
  ${TEB_PREFIX}/include/suitesparse
)
target_link_directories(pyteb PRIVATE ${TEB_PREFIX}/lib)
target_link_libraries(pyteb PRIVATE
  teb_local_planner
  g2o_core g2o_stuff g2o_types_slam2d
  g2o_solver_csparse g2o_csparse_extension
  roscpp rosconsole rosconsole_backend_interface roscpp_serialization rostime cpp_common
)
target_compile_definitions(pyteb PRIVATE
  _USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN
  BOOST_ALL_NO_LIB _CRT_SECURE_NO_WARNINGS
  # RoboStack builds the ROS libs as DLLs; without this the ROS headers declare
  # exported *data* symbols (ros::console::g_initialized) without dllimport and
  # the link fails with LNK2001.
  ROS_BUILD_SHARED_LIBS
)
target_compile_options(pyteb PRIVATE /bigobj /EHsc /wd4996 /wd4267 /wd4244)
