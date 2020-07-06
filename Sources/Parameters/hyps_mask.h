#ifndef HYP_MASK_H
#define HYP_MASK_H

#include <vector>
#include <unordered_map>
#include <string>

class HypsMask{
    public:
        HypsMask();

        // Base attributes on the as_dict method of parameter_helper.py
        int nspecie = 1, ntwobody = 1, nthreebody = 1, nmanybody = 1,
            twobody_start, threebody_start, manybody_start, ncut3b = 0;
        std::vector<int> specie_mask, twobody_mask, threebody_mask,
            manybody_mask, cut3b_mask, map;
        std::vector<double> twobody_cutoff_list, threebody_cutoff_list,
            manybody_cutoff_list, original_hyps, hyps;
        std::vector<std::string> original_labels, kernels, hyp_labels;
        bool train_noise;
        double energy_noise;
        std::string kernel_name;
        std::unordered_map<std::string, double> cutoffs;

};

#endif
