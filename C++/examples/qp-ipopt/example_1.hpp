#ifndef QCQP_1_H
#define QCQP_1_H

#include "NLP/QP.hpp"

const size_t NX = 2;
const size_t NG = 2;
const size_t NH = 0;

class example_1 : public NLP<double, NX, NG, NH> {
    public:

        example_1() = default;
        virtual ~example_1() = default;

        virtual const double objectiveFunction(const Eigen::Matrix<double, NX, 1>& candidate_vector) const override {
            return candidate_vector[1]*(5+candidate_vector[0]);
        }
        virtual const Eigen::Matrix<double, NX, 1> gradientFunction(const Eigen::Matrix<double, NX, 1>& candidate_vector) const override {
            return {
                    candidate_vector[1], 
                    5+candidate_vector[0],
            };
        }
        virtual const Eigen::Matrix<double, NX, NX> hessianFunction(const Eigen::Matrix<double, NX, 1>& candidate_vector) const override {
            Eigen::Matrix<double, NX, NX> hessian = Eigen::Matrix<double, NX, NX>({{0., 1.}, {1., 0.}});
            return hessian;
        }

        virtual const std::optional<Eigen::Matrix<double, NG, 1>> inequalityConstraintVector(const Eigen::Matrix<double, NX, 1>& candidate_vector) const override {
            Eigen::Matrix<double, NG, 1> g;
            g(0, 0) = candidate_vector[0]*candidate_vector[1]-5.;
            g(1, 0) = -(std::pow(candidate_vector[0], 2) + std::pow(candidate_vector[1], 2) - 20.);

            return g;
        };
        virtual const std::optional<Eigen::Matrix<double, NX, NG>> gradientInequalityConstraintVector(const Eigen::Matrix<double, NX, 1>& candidate_vector) const override {
            Eigen::Matrix<double, NX, NG> d_g = Eigen::Matrix<double, NX, NG>::Zero();

            d_g(0, 0) = candidate_vector[1];
            d_g(1, 0) = candidate_vector[0];
            d_g(0, 1) = -2*candidate_vector[0];
            d_g(1, 1) = -2*candidate_vector[1];

            return d_g;
        };
};

#endif // QCQP_1_H