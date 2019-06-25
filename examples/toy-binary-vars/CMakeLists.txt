add_executable(run_fembv_bin
    run_fembv_bin.cpp)

target_link_libraries(run_fembv_bin
    FEMBVPP::fembvpp)

if(ENABLE_COMPILER_WARNINGS)
    target_compile_options(run_fembv_bin
        PRIVATE "${CXX_WARNING_FLAGS}")
endif()

set_target_properties(run_fembv_bin PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS OFF)

add_executable(FEMBVPP::run_fembv_bin ALIAS run_fembv_bin)

add_dependencies(examples run_fembv_bin)
