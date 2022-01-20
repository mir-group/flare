# make executable
add_executable(time time_single_bond.cpp)
target_include_directories(time PUBLIC ${ACE_INCLUDE_DIR})
target_link_libraries(time PUBLIC flare_pp)


add_executable(benchmark benchmark_B2.cpp)
target_include_directories(benchmark PUBLIC ${ACE_INCLUDE_DIR})
target_link_libraries(benchmark PUBLIC flare_pp)
