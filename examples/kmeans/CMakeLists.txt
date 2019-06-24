add_executable(run_fembv_kmeans
    multivariate_normal.cpp
    run_fembv_kmeans.cpp)

target_link_libraries(run_fembv_kmeans
    FEMBVPP::fembvpp
)

if(ENABLE_COMPILER_WARNINGS)
    target_compile_options(run_fembv_kmeans
        PRIVATE "${CXX_WARNING_FLAGS}")
endif()

add_executable(FEMBVPP::run_fembv_kmeans ALIAS run_fembv_kmeans)

add_dependencies(examples run_fembv_kmeans)
