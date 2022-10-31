# make executable
add_executable(
  tests
  test_y_grad.cpp
  test_radial.cpp
  test_structure.cpp
  test_n_body.cpp
  test_sparse_gp.cpp
  test_descriptor.cpp
  test_kernels.cpp
  test_json.cpp
)

include_directories(../src/flare_pp)
if(EXIST_GTEST)
    target_link_libraries(tests GTest::gtest GTest::gtest_main flare)
else()
    target_link_libraries(tests PUBLIC gtest gtest_main flare)
endif()
