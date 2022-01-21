#include <chrono>
#include <iostream>
#include <cmath>

#include <Eigen/Dense>

#include "structure.h"
#include "local_environment.h"


void construct_structure(Structure** struc) {
    int noa = 50;
    int nos = 5;

    std::vector<int> species = std::vector<int>(noa);
    for (int i = 0; i < noa; i++) {
        species[i] = rand() % nos;
    }

    Eigen::MatrixXd cell {3, 3};
    Eigen::MatrixXd positions_1 = 10 * Eigen::MatrixXd::Random(noa, 3);


    cell << 10, 0.52, 0.12,
           -0.93, 10, 0.32,
            0.1, -0.2, 10;

    (*struc) = new Structure(cell, species, positions_1);
}

void deconstruct_structure(Structure* struc) {
    delete struc;
}

void construct_environment(LocalEnvironment** env, Structure* struc) {
    double rcut = 10;
    (*env) = new LocalEnvironment(*struc, 0, rcut);
}

void deconstruct_environment(LocalEnvironment* env) {
    delete env;
}

void construct_calculator(B2_Calculator** calc) {
    std::string radial_string = "chebyshev";
    std::string cutoff_string = "quadratic";

    std::vector<double> cutoff_hyps;

    double first_gauss = 0;
    double final_gauss = 10;
    std::vector<double> radial_hyps = {first_gauss, final_gauss};

    std::vector<int> descriptor_settings {5, 10, 10};

    (*calc) = new B2_Calculator(radial_string, cutoff_string, radial_hyps,
                            cutoff_hyps, descriptor_settings, 0);
}

void deconstruct_calculator(B2_Calculator* calc) {
    delete calc;
}

int main() {
    Structure* struc;
    LocalEnvironment* env;
    B2_Calculator* calc;
    double duration = 0;
    int num_reps = 20;

    construct_calculator(&calc);


    for (int i = 0; i < num_reps; i++) {
        construct_structure(&struc);
        construct_environment(&env, struc);

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        calc->compute(*env);

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

        deconstruct_structure(struc);
        deconstruct_environment(env);
    }


    std::cout << "time: " << duration << " ms" << std::endl;

    deconstruct_calculator(calc);
    return 0;
}
