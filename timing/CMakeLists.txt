# make executable
add_executable(time time_single_bond.cpp)
target_include_directories(time PUBLIC ${ACE_INCLUDE_DIR})
target_link_libraries(time PUBLIC Ace)
