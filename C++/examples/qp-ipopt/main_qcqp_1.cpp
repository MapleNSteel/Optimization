#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <chrono>

#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "QCQP_1.hpp"
#include "Solver/IPOPT.hpp"

int main() {

    const size_t NUM_ITER = 100;

    const Eigen::Matrix<double, NX, NX> Q = Eigen::DiagonalMatrix<double, NX>(Eigen::Matrix<double, NX, 1>::Ones());
    const Eigen::Matrix<double, NX, 1> g = Eigen::Matrix<double, NX, 1>::Zero();

    const QCQP_1 qcqp_1(Q, g);

    double mu = 0.8;

    IPOPT<double, NX, NG, NH> ipopt(qcqp_1, mu);
    
    Eigen::Matrix<double, NX, 1> x_soln = (Eigen::Matrix<double, NX, 1>::Ones()*1.5).eval();
    Eigen::Matrix<double, NG, 1> sigma = mu*qcqp_1.inequalityConstraintVector(x_soln).value().cwiseInverse();

    std::cout << "Initial sigma :\n" << sigma << "\n";

    for (size_t num = 0; num < NUM_ITER; num++) {
        const Eigen::Matrix<double, NX+NG+NH, 1> x_opt = ipopt.solve(x_soln, sigma);

        x_soln.block<NX, 1>(0, 0) = x_opt.block<NX, 1>(0, 0).eval();
        sigma.block<NG, 1>(0, 0) = x_opt.block<NG, 1>(NX, 0).eval();

        mu *= 0.8;

        ipopt.updateMu(mu);
    }

    std::cout << "Objective function: " << qcqp_1.objectiveFunction(x_soln) << std::endl;
    std::cout << "Optimized solution:\n" << x_soln << std::endl;
    std::cout << "Sigma:\n" << sigma << std::endl;
    std::cout << "Constraint vector:\n" << qcqp_1.inequalityConstraintVector(x_soln).value().cwiseInverse() << std::endl;

    return 0;
}
