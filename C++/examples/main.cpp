#include <functional>
#include <iostream>
#include <math.h>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "NLP/QP.hpp"
#include "Solver/IPOPT.hpp"

int main() {
    std::function<double(Eigen::VectorXd)> f = [](Eigen::VectorXd x) -> double {
        return std::pow(x[0], 2) + std::pow(x[1], 2);
    };
    std::function<Eigen::VectorXd(Eigen::VectorXd)> df = [](Eigen::VectorXd x) -> Eigen::VectorXd {
        return Eigen::VectorXd({{2 * x[0], 2 * x[1]}});
    };

    const QP quadratic_program;

    const IPOPT<double, 2, 0, 0> solver(nlp);
    
    Eigen::Matrix<double, 2, 1> x0 = {1.0, 1.0};
    Eigen::Matrix<double, 2, 1> x_opt = solver.solve(x0);
    std::cout << "Optimized solution: (" << x_opt[0] << ", " << x_opt[1] << ")" << std::endl;

    return 0;
}
