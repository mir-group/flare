#ifndef JSON_H
#define JSON_H

#include <nlohmann/json.hpp>

// Cf. the Simox library on Gitlab.
// TODO: Use templating to reduce redundancy.
namespace nlohmann {
  template <> struct adl_serializer<Eigen::VectorXi> {
    static void to_json(json &j, const Eigen::VectorXi &vector) {
      for (int row = 0; row < vector.size(); row++) {
        j.push_back(vector(row));
      }
    }

    static void from_json(const json &j, Eigen::VectorXi &vector) {
      int n_rows = j.size();
      vector = Eigen::VectorXi::Zero(n_rows);

      for (int row = 0; row < j.size(); row++) {
        const auto &value = j.at(row);
        vector(row) = value.get<int>();
      }
    }
  };

  template <> struct adl_serializer<Eigen::VectorXd> {
    static void to_json(json &j, const Eigen::VectorXd &vector) {
      for (int row = 0; row < vector.size(); row++) {
        j.push_back(vector(row));
      }
    }

    static void from_json(const json &j, Eigen::VectorXd &vector) {
      int n_rows = j.size();
      vector = Eigen::VectorXd::Zero(n_rows);

      for (int row = 0; row < j.size(); row++) {
        const auto &value = j.at(row);
        vector(row) = value.get<double>();
      }
    }
  };

  template <> struct adl_serializer<Eigen::MatrixXd> {
    static void to_json(json &j, const Eigen::MatrixXd &matrix) {
      for (int row = 0; row < matrix.rows(); row++) {
        nlohmann::json column = nlohmann::json::array();

        for (int col = 0; col < matrix.cols(); col++) {
          column.push_back(matrix(row, col));
        }

        j.push_back(column);
      }
    }

    static void from_json(const json &j, Eigen::MatrixXd &matrix) {
      int n_rows = j.size();
      int n_cols;
      if (n_rows > 0)
        n_cols = j.at(0).size();
      else
        n_cols = 0;
      matrix = Eigen::MatrixXd::Zero(n_rows, n_cols);

      for (int row = 0; row < j.size(); row++) {
        const auto &jrow = j.at(row);
        for (int col = 0; col < jrow.size(); col++) {
          const auto &value = jrow.at(col);
          matrix(row, col) = value.get<double>();
        }
      }
    }
  };
}

#endif
