#include "ace.h"
#include <cmath>
#define Pi 3.14159265358979323846

void quadratic_cutoff(double * rcut_vals, double r, double rcut,
                      std::vector<double> cutoff_hyps){
                
    if (r > rcut){
        rcut_vals[0] = 0;
        rcut_vals[1] = 0;
        return;
    }

    double rdiff = r - rcut;
    rcut_vals[0] = rdiff * rdiff;
    rcut_vals[1] = 2 * rdiff;
}

void cos_cutoff(double * rcut_vals, double r, double rcut,
                std::vector<double> cutoff_hyps){

    // Calculate the cosine cutoff function and its gradient. If r > rcut, the array is returned unchanged.
    if (r > rcut){
        rcut_vals[0] = 0;
        rcut_vals[1] = 0;
        return;
    }

    double cutoff_val = (1./2.) * (cos(Pi * r / rcut) + 1);
    double cutoff_derv = -Pi * sin(Pi * r / rcut) / (2 * rcut);

    rcut_vals[0] = cutoff_val;
    rcut_vals[1] = cutoff_derv;
}

void hard_cutoff(double * rcut_vals, double r, double rcut,
                 std::vector<double> cutoff_hyps){
    if (r > rcut){
        rcut_vals[0] = 0;
        rcut_vals[1] = 0;
        return;
    }

    rcut_vals[0] = 1;
    rcut_vals[1] = 0;

}
