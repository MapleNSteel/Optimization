#ifndef LCQP_1_H
#define LCQP_1_H

#include "NLP/QP.hpp"

const size_t NX = 2;
const size_t NG = 2;
const size_t NH = 1;

class LCQP_1 : public QP<double, NX, NG, NH> {
    public:

        LCQP_1(const Eigen::Matrix<double, NX, NX>& Q,
               const Eigen::Matrix<double, NX, 1>& g,
               const std::optional<Eigen::Matrix<double, NG, NX>> G = std::nullopt,
               const std::optional<Eigen::Matrix<double, NG, 1>> h = std::nullopt)
            :  QP(Q, g) 
        {}
        virtual ~LCQP_1() = default;

        virtual const std::optional<Eigen::Matrix<double, NG, 1>> inequalityConstraintVector(const Eigen::Matrix<double, NX, 1>& candidate_vector) const {
            Eigen::Matrix<double, NG, 1> g{
                candidate_vector[0] - 1,
                candidate_vector[1] - 1
            };

            return g;
        };
        virtual const std::optional<Eigen::Matrix<double, NX, NG>> gradientInequalityConstraintVector(const Eigen::Matrix<double, NX, 1>& candidate_vector) const {
            Eigen::Matrix<double, NX, NG> d_g = Eigen::Matrix<double, NX, NG>::Zero();

            d_g(0, 0) = 1;
            d_g(1, 1) = 1;

            return d_g;
        };

        virtual const std::optional<Eigen::Matrix<double, NH, 1>> equalityConstraintVector(const Eigen::Matrix<double, NX, 1>& candidate_vector) const {
            Eigen::Matrix<double, NH, 1> h{
                candidate_vector[0] + candidate_vector[1] - 3.,
            };

            return h;
        };
        virtual const std::optional<Eigen::Matrix<double, NX, NH>> gradientEqualityConstraintVector(const Eigen::Matrix<double, NX, 1>& candidate_vector) const {
            Eigen::Matrix<double, NX, NH> d_h = Eigen::Matrix<double, NX, NH>::Zero();

            d_h(0, 0) = 1;
            d_h(1, 0) = 1;

            return d_h;
        };
};

#endif // LCQP_1_H