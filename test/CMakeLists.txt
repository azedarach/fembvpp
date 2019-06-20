add_executable(fembvpp_unit
    catch_main.cpp
    test_clpsimplex_affiliations_solver.cpp)

target_include_directories(fembvpp_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/catch
    ${EIGEN3_INCLUDE_DIR})

if(ENABLE_COMPILER_WARNINGS)
    target_compile_options(fembvpp_unit PRIVATE
        ${CXX_WARNING_FLAGS})
endif()

target_link_libraries(fembvpp_unit PRIVATE
    FEMBVPP::fembvpp)

include(ParseAndAddCatchTests)
ParseAndAddCatchTests(fembvpp_unit)

add_executable(FEMBVPP::fembvpp_unit ALIAS fembvpp_unit)

add_custom_target(check COMMAND "ctest")
add_dependencies(check fembvpp_unit)
