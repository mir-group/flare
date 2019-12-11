#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

TEST(StructureTest, EigenTest){
    MatrixXd m = MatrixXd::Random(3,3);
    cout << "let's take the inverse." << endl;
    MatrixXd m_inv = m.inverse();
    cout << m_inv << endl;
    cout <<  "let's try matrix multiplication." << endl;
    cout << m * m_inv << endl;
    EXPECT_EQ(1, 1);
}

// TEST(StructureTest, ConstructorTest){
//     vector<double> xs {1, 2, 3};
//     vector<double> ys {4, 5, 6};
//     vector<double> zs {7, 8, 9};
//     vector<double> vec1 {1, 0, 0};
//     vector<double> vec2 {0, 1, 0};
//     vector<double> vec3 {0, 0, 1};
//     vector<int> species {1, 2, 3};
//     Structure test = Structure(xs, ys, zs, vec1, vec2, vec3, species);
//     cout << test.xs[2];
// }
