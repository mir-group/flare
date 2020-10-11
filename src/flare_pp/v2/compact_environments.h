#ifndef COMPACT_ENVIRONMENTS_H
#define COMPACT_ENVIRONMENTS_H

#include <Eigen/Dense>
#include <vector>
class CompactStructure;

class CompactEnvironments {
public:
  CompactEnvironments();

  std::vector<Eigen::MatrixXd> descriptors;
  std::vector<std::vector<double>> descriptor_norms;
  std::vector<int> n_atoms, c_atoms;
  int n_species, n_descriptors;
  int n_envs = 0;

  void add_environments(const CompactStructure &structure,
                        std::vector<int> environments);
};

#endif
