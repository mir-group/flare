#include "ace.h"
using namespace std;

Structure :: Structure(const vector<double> & xs, const vector<double> & ys,
                       const vector<double> & zs, const vector<double> & vec1,
                       const vector<double> & vec2,
                       const vector<double> & vec3,
                       const vector<int> & species){
    this->xs = xs;
    this->ys = ys;
    this->zs = zs;
    this->vec1 = vec1;
    this->vec2 = vec2;
    this->vec3 = vec3;
    this->species = species;
}
