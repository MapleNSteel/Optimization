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
    virtual const Eigen::Matrix<T, NX+NG+NH, 1> solve(const Eigen::Matrix<T, NX, 1>& candidate_vector, const std::optional<Eigen::Matrix<T, NG, 1>>& sigma_vector = std::nullopt, const std::optional<Eigen::Matrix<T, NH, 1>>& lambda_vector = std::nullopt) const override {

      const Eigen::Matrix<T, NX, 1> df_dx = this->m_nlp.gradientFunction(candidate_vector);

      Eigen::Matrix<T, NX+NG+NH, NX+NG+NH> G;
      Eigen::Matrix<T, NX+NG+NH, 1> b;

      Eigen::Matrix<T, NX+NG+NH, 1> combined_candidate;

      const Eigen::Matrix<T, NX, NX> A = this->m_nlp.hessianFunction(candidate_vector);
      G.block(0, 0, NX, NX) = std::move(A);
      b.block(0, 0, NX, 1) = -df_dx;

      combined_candidate.block(0, 0, NX, 1) = candidate_vector;

      if (sigma_vector) {
        const Eigen::Matrix<T, NX, NG> dg_dx = this->m_nlp.gradientInequalityConstraintVector(candidate_vector).value();

        const Eigen::Matrix<T, NX, NG> B = dg_dx;
        const Eigen::Matrix<T, NG, NX> C = (sigma_vector.value().asDiagonal()*dg_dx.transpose()).eval();
        const Eigen::Matrix<T, NG, NG> D = Eigen::DiagonalMatrix<T, NG>(this->m_nlp.inequalityConstraintVector(candidate_vector).value());

        G.block(0, NX, NX, NG) = std::move(B);
        G.block(NX, 0, NG, NX) = std::move(C);
        G.block(NX, NX, NG, NG) = std::move(D);

        b.block(0, 0, NX, 1).noalias() += -dg_dx*sigma_vector.value();
        b.block(NX, 0, NG, 1) = -(D*sigma_vector.value() + this->m_mu*Eigen::Matrix<T, NG, 1>::Ones()).eval();
        
        combined_candidate.block(NX, 0, NG, 1) = sigma_vector.value();
      }
      if (lambda_vector) {
        const Eigen::Matrix<T, NX, NH> dh_dx = this->m_nlp.gradientEqualityConstraintVector(candidate_vector).value();

        G.block(0, NX+NG, NX, NH) = dh_dx;
        G.block(NX+NG, 0, NH, NX) = dh_dx.transpose();

        b.block(0, 0, NX, 1).noalias() += -dh_dx*lambda_vector.value();
        b.block(NX+NG, 0, NH, 1) = -this->m_nlp.equalityConstraintVector(candidate_vector).value();
        
        combined_candidate.block(NX+NG, 0, NH, 1) = lambda_vector.value();
      }

      Eigen::Matrix<T, NX+NG+NH, 1> solution =(combined_candidate + G.fullPivLu().solve(b)).eval();

      return solution;
    };

private:
    double m_mu;
};

#endif // IPOPT_H
