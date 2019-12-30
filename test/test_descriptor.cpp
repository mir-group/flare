#include "gtest/gtest.h"
#include "ace.h"
#include <iostream>
#include <cmath>

using namespace std;

TEST(DummyTest, DummyTest){
    vector<vector<double>> test = 
        vector<vector<double>> {};
    
    for (int n = 0; n < 5; n ++){
        test.push_back({});
    }
    test[0].push_back({1});
    // test.push_back(vector<double> {1, 2, 3});

    cout << test[1].size() << endl;
}
