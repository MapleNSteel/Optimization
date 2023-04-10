#include <chrono>

#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "LCQP_1.hpp"
#include "Solver/IPOPT.hpp"

int main() {

    const size_t NUM_ITER = 100;

    const Eigen::Matrix<double, NX, NX> Q = Eigen::DiagonalMatrix<double, NX>(Eigen::Matrix<double, NX, 1>::Ones());
    const Eigen::Matrix<double, NX, 1> g = Eigen::Matrix<double, NX, 1>::Zero();

    const LCQP_1 lcqp_1(Q, g);

    double mu = 0.8;

    IPOPT<double, NX, NG, NH> ipopt(lcqp_1, mu);
    
    Eigen::Matrix<double, NX, 1> x0 = (Eigen::Matrix<double, NX, 1>::Ones()*3.).eval();
    Eigen::Matrix<double, NG, 1> sigma = mu*lcqp_1.inequalityConstraintVector(x0).value().cwiseInverse();
    // Eigen::Matrix<double, NH, 1> lambda = Eigen::Matrix<double, NH, 1>::Ones();

    for (size_t num = 0; num < NUM_ITER; num++) {
        const Eigen::Matrix<double, NX+NG+NH, 1> x_opt = ipopt.solve(x0, sigma);

        x0.block<NX, 1>(0, 0) = x_opt.block<NX, 1>(0, 0).eval();
        sigma.block<NG, 1>(0, 0) = x_opt.block<NG, 1>(NX, 0).eval();

        mu *= 0.8;

        ipopt.updateMu(mu);
    }

    std::cout << "Objective function: " << lcqp_1.objectiveFunction(x0) << std::endl;
    std::cout << "Optimized solution:\n" << x0 << std::endl;
    std::cout << "Sigma:\n" << sigma << std::endl;

    return 0;
}
