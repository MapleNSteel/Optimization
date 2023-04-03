#ifndef IPOPT_H
#define IPOPT_H

#include <functional>
#include <vector>

#include "Solver.hpp"

/**
 * @brief An implementation of IPOPT
 * 
 * @tparam The type used for the optimization
 */
template <typename T, size_t NX, size_t NG=0, size_t NH=0>
class IPOPT : public Solver<T, NX, NG, NH> {
public:
    /**
     * @brief Construct a new IPOPT object.
     * 
     * @param nlp Underlying NLP to solved@param mu Mu for the slackness of the constrasize_ts.
     * @param tolerance Tolerance for convergence.
     * @param max_iterations Maximum number of iterations.
     */
    IPOPT(NLP<T, NX, NG, NH>& nlp,
          double mu = 1.,
          double tolerance = 1e-5,
          size_t max_iterations = 1000)
        : Solver<double, NX, NG, NH>(nlp, tolerance, max_iterations),
          m_mu(mu)
    {}
    
    /**
     * @brief Function to solve the objective function using IPOPT.
     * 
     * @param initial_solution_vector The initial solution vector.
     * @return std::vector<T> The solution vector.
     */
    virtual Eigen::Matrix<T, NX, 1> const solve(Eigen::Matrix<T, NX, 1> initial_solution_vector) override {
      // TODO: Implement IPOPT solver
      return initial_solution_vector; // temporary placeholder
    };

private:
    double m_mu;
};

#endif // IPOPT_H
