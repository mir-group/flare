#include <cmath>
#define Pi 3.14159265358979323846

void cos_cutoff(double * rcut_vals, double r, double rcut){
    // Calculate the cosine cutoff function and its gradient. Assumes that the input is an array of four zeros. If r > rcut, the array is returned unchanged.

    if (r > rcut){
        return;
    }

    double cutoff_val = (1./2.) * (cos(Pi * r / rcut) + 1);
    double cutoff_derv = -Pi * sin(Pi * r / rcut) / (2 * rcut);

    rcut_vals[0] = cutoff_val;
    rcut_vals[1] = cutoff_derv;
}
