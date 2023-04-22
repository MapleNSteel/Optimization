#ifndef QCQP_1_H
#define QCQP_1_H

#include "NLP/QP.hpp"

const size_t NX = 2;
const size_t NG = 1;
const size_t NH = 0;

class QCQP_1 : public QP<double, NX, NG, NH> {
    public:

        QCQP_1(const Eigen::Matrix<double, NX, NX>& Q,
               const Eigen::Matrix<double, NX, 1>& g,
               const std::optional<Eigen::Matrix<double, NG, NX>> G = std::nullopt,
               const std::optional<Eigen::Matrix<double, NG, 1>> h = std::nullopt)
            :  QP(Q, g) 
        {}
        virtual ~QCQP_1() = default;

        virtual const std::optional<Eigen::Matrix<double, NG, 1>> inequalityConstraintVector(const Eigen::Matrix<double, NX, 1>& candidate_vector) const {
            Eigen::Matrix<double, NG, 1> g{
                -(std::pow(candidate_vector[0]-1, 2) + std::pow(candidate_vector[1]-1, 2) - 1.),
            };

            return g;
        };
        virtual const std::optional<Eigen::Matrix<double, NX, NG>> gradientInequalityConstraintVector(const Eigen::Matrix<double, NX, 1>& candidate_vector) const {
            Eigen::Matrix<double, NX, NG> d_g = Eigen::Matrix<double, NX, NG>::Zero();

            d_g(0, 0) = -2*(candidate_vector[0]-1);
            d_g(1, 0) = -2*(candidate_vector[1]-1);

            return d_g;
        };
};

#endif // QCQP_1_H