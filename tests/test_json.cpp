#include <nlohmann/json.hpp>
#include <iostream>
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include "json.h"

TEST(JsonTest, MatTest){
  Eigen::MatrixXd test = Eigen::MatrixXd::Random(3, 3);

  nlohmann::json j = test;
  Eigen::MatrixXd test2 = j;

  for (int i = 0; i < test.rows(); i++){
    for (int j = 0; j < test.cols(); j++){
      EXPECT_EQ(test(i, j), test2(i, j));
    }
  }
};

TEST(JsonTest, SizeTest){
  nlohmann::json j;
  std::cout << j.size() << std::endl;

  int test = 1;
  j.push_back({"test", test});
  j.push_back({"test2", test});

  std::cout << j.size() << std::endl;
  std::cout << j[0] << std::endl;
  std::cout << j[1] << std::endl;
}

