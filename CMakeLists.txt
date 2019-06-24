cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)

project(FEMBVPP VERSION 0.0.1 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include(FEMBVPPBuildOptions)

if(NOT FEMBVPP_INSTALL_INCLUDE_DIR)
    set(FEMBVPP_INSTALL_INCLUDE_DIR include/fembvpp
        CACHE PATH
        "The directory relative to CMAKE_INSTALL_PREFIX where FEMBVPP headers are installed")
endif()

if(NOT FEMBVPP_INSTALL_ARCHIVE_DIR)
    set(FEMBVPP_INSTALL_ARCHIVE_DIR lib
        CACHE PATH
        "The directory relative to CMAKE_INSTALL_PREFIX where FEMBVPP libraries are installed")
endif()

if(NOT FEMBVPP_INSTALL_LIBRARY_DIR)
    set(FEMBVPP_INSTALL_LIBRARY_DIR lib
        CACHE PATH
        "The directory relative to CMAKE_INSTALL_PREFIX where FEMBVPP libraries are installed")
endif()

if(NOT FEMBVPP_INSTALL_CMAKE_DIR)
    set(FEMBVPP_INSTALL_CMAKE_DIR share/fembvpp
        CACHE PATH
        "The directory relative to CMAKE_INSTALL_PREFIX where FEMBVPP CMake files are installed")
endif()

if(ENABLE_COMPILER_WARNINGS)
    include(add_cxx_flag_if_supported)
    include(CompilerWarnings)
endif()

find_package(CoinUtils REQUIRED)
find_package(Clp REQUIRED)
find_package(Eigen3 3.1 REQUIRED)
find_package(NLopt REQUIRED)

set(FEMBVPP_PUBLIC_HEADERS
    include/fembvpp/clpsimplex_affiliations_solver.hpp
    include/fembvpp/fembv_bin.hpp
    include/fembvpp/fembv_bin_local_model.hpp
    include/fembvpp/fembv_kmeans.hpp
    include/fembvpp/linear_constraint.hpp)

add_library(fembvpp
    src/clpsimplex_affiliations_solver.cpp
    src/fembv_bin.cpp
    src/fembv_bin_local_model.cpp
    src/fembv_kmeans.cpp
    src/linear_constraint.cpp)

target_include_directories(fembvpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include/fembvpp
    $<INSTALL_INTERFACE:{FEMBVPP_INSTALL_INCLUDE_DIR}>
    ${EIGEN3_INCLUDE_DIR})

target_link_libraries(fembvpp PUBLIC
    Clp::Clp
    CoinUtils::CoinUtils
    NLopt::NLopt)

if(ENABLE_COMPILER_WARNINGS)
    target_compile_options(fembvpp PRIVATE
        ${CXX_WARNING_FLAGS})
endif()

set_target_properties(fembvpp PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS OFF
    PUBLIC_HEADER
    "${FEMBVPP_PUBLIC_HEADERS}")

install(TARGETS
    fembvpp
    EXPORT FEMBVPPTargets
    ARCHIVE DESTINATION ${FEMBVPP_INSTALL_ARCHIVE_DIR}
    PUBLIC_HEADER DESTINATION ${FEMBVPP_INSTALL_INCLUDE_DIR})

add_library(FEMBVPP::fembvpp ALIAS fembvpp)

export(TARGETS fembvpp
    NAMESPACE FEMBVPP::
    FILE "${PROJECT_BINARY_DIR}/${FEMBVPP_INSTALL_CMAKE_DIR}/FEMBVPPTargets.cmake")

export(PACKAGE FEMBVPP)

add_subdirectory(examples EXCLUDE_FROM_ALL)

if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(test EXCLUDE_FROM_ALL)
endif()

include(FEMBVPPGenerateConfig)
generate_fembvpp_config_files()
