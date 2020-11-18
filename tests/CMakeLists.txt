# make executable
add_executable(
  tests
  test_y_grad.cpp
  test_radial.cpp
  test_structure.cpp
  test_n_body.cpp
  test_sparse_gp_dtc.cpp
  test_descriptor.cpp
)

include_directories(../src/flare_pp)
target_link_libraries(tests PUBLIC gtest gtest_main flare)
