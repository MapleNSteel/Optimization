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
    IPOPT(const NLP<T, NX, NG, NH>& nlp,
          T mu = 1.,
          T tolerance = 1e-5,
          size_t max_iterations = 1000)
        : Solver<double, NX, NG, NH>(nlp, tolerance, max_iterations),
          m_mu(std::move(mu))
    {}

    void updateMu(const T& mu) {
      this->m_mu = mu;
    }
    
    /**
     * @brief Function to solve the objective function using IPOPT.
     * 
     * @param initial_solution_vector The initial solution vector.
     * @return std::vector<T> The solution vector.
     */
    virtual const Eigen::Matrix<T, NX+NG, 1> solve(const Eigen::Matrix<T, NX, 1>& candidate_vector, const std::optional<Eigen::Matrix<T, NG, 1>>& lambda_vector = std::nullopt) const override {

      const Eigen::Matrix<T, NX, 1> df_dx = std::move(this->m_nlp.gradientFunction(candidate_vector));
      const Eigen::Matrix<T, NX, NG> dg_dx = this->m_nlp.gradientInequalityConstraintVector(candidate_vector).value();

      const Eigen::Matrix<T, NX, NX> A = std::move(this->m_nlp.hessianFunction(candidate_vector));
      const Eigen::Matrix<T, NX, NG> B = -dg_dx;
      const Eigen::Matrix<T, NG, NX> C = (lambda_vector.value().asDiagonal()*dg_dx.transpose()).eval();
      const Eigen::Matrix<T, NG, NG> D = std::move(Eigen::DiagonalMatrix<T, NG>(this->m_nlp.inequalityConstraintVector(candidate_vector).value()));

      Eigen::Matrix<T, NX+NG, NX+NG> G;
      Eigen::Matrix<T, NX+NG, 1> b;

      G.block(0, 0, NX, NX).noalias() = A;
      G.block(0, NX, NX, NG).noalias() = B;
      G.block(NX, 0, NG, NX).noalias() = C;
      G.block(NX, NX, NG, NG).noalias() = D;

      b.block(0, 0, NX, 1).noalias() = -(df_dx - dg_dx*lambda_vector.value()).eval();
      b.block(NX, 0, NG, 1).noalias() = -(D*lambda_vector.value() - this->m_mu*Eigen::Matrix<T, NG, 1>::Ones()).eval();

      Eigen::Matrix<T, NX+NG, 1> combined_candidate;

      combined_candidate.block(0, 0, NX, 1).noalias() = candidate_vector;
      combined_candidate.block(NX, 0, NG, 1).noalias() = lambda_vector.value();

      Eigen::Matrix<T, NX+NG, 1> solution =(combined_candidate + G.fullPivLu().solve(b)).eval();

      return solution;
    };

private:
    double m_mu;
};

#endif // IPOPT_H
