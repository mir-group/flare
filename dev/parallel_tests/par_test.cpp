#include <iostream>
#include <thread> 
#include <chrono>
#include <vector>
#include <Eigen/Dense>

int main(){
    auto t1 = std::chrono::high_resolution_clock::now();

    int mat_size = 5;
    Eigen::MatrixXd kern_mat = Eigen::MatrixXd::Zero(mat_size, mat_size);
    Eigen::VectorXd vec_curr(5);

    #pragma omp parallel for
    for (int i = 0; i < mat_size; i ++){
        std::this_thread::sleep_for(std::chrono::seconds(1));
        vec_curr << i, i+1, i+2, i+3, i+4;
        kern_mat.col(i) = vec_curr;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto tot_time = 
        std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    std::cout << tot_time << std::endl;
    std::cout << kern_mat << std::endl;

}
