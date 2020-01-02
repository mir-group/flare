#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <cmath>

using namespace std;

TEST(DummyTest, DummyTest){
    int test1 = 5;
    int & test2 = test1;

    test2 ++;

    // std::cout << test1;
    // std::cout << test2;
}
