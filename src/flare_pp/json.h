#include <nlohmann/json.hpp>

namespace nlohmann {
    template <>
    struct adl_serializer<Eigen::VectorXi> {
        static void to_json(json& j, const Eigen::VectorXi& opt) {

        }

        static void from_json(const json& j, Eigen::VectorXi& opt) {

        }
    };

    template <>
    struct adl_serializer<Eigen::VectorXd> {
        static void to_json(json& j, const Eigen::VectorXd& opt) {

        }

        static void from_json(const json& j, Eigen::VectorXd& opt) {

        }
    };

    template <>
    struct adl_serializer<Eigen::MatrixXd> {
        static void to_json(json& j, const Eigen::MatrixXd& opt) {

        }

        static void from_json(const json& j, Eigen::MatrixXd& opt) {

        }
    };
}
