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
          m_mu(mu)
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
    virtual const Eigen::Matrix<T, NX+NH, 1> solve(const Eigen::Matrix<T, NX, 1>& candidate_vector, const std::optional<Eigen::Matrix<T, NH, 1>>& sigma_vector = std::nullopt) const override {

      const std::optional<Eigen::Matrix<T, NH, 1>> h = this->m_nlp.equalityConstraintVector(candidate_vector);
      const std::optional<Eigen::Matrix<T, NG, 1>> g = this->m_nlp.inequalityConstraintVector(candidate_vector);

      const Eigen::Matrix<T, NX, 1> df_dx = this->m_nlp.gradientFunction(candidate_vector);
      const std::optional<Eigen::Matrix<T, NX, NH>> dh_dx = this->m_nlp.gradientEqualityConstraintVector(candidate_vector);
      const std::optional<Eigen::Matrix<T, NX, NG>> dg_dx = this->m_nlp.gradientInequalityConstraintVector(candidate_vector);

      Eigen::Matrix<T, NG, NX> L;

      Eigen::Matrix<T, NX+NH, NX+NH> G = Eigen::Matrix<double, NX+NH, NX+NH>::Zero();
      Eigen::Matrix<T, NX+NH, 1> b = Eigen::Matrix<double, NX+NH, 1>::Zero();

      Eigen::Matrix<T, NX+NH, 1> combined_candidate;

      const Eigen::Matrix<T, NX, NX> A = this->m_nlp.hessianFunction(candidate_vector);
      G.block(0, 0, NX, NX) = A;
      b.block(0, 0, NX, 1) = -df_dx;

      combined_candidate.block(0, 0, NX, 1) = candidate_vector;

      T alpha = 1.;

      if (NH) {
        G.block(0, NX, NX, NH) = dh_dx.value();
        G.block(NX, 0, NH, NX) = dh_dx.value().transpose();

        b.block(0, 0, NX, 1).noalias() += -dh_dx.value()*sigma_vector.value();
        b.block(NX, 0, NH, 1) = -h.value();
        
        combined_candidate.block(NX, 0, NH, 1) = sigma_vector.value();
      }

      if (NG) {
        const std::optional<Eigen::Matrix<T, NG, 1>> lambda_vector = this->m_mu*g.value().cwiseInverse();

        L = this->m_mu*Eigen::DiagonalMatrix<T, NG>(g.value().cwiseAbs2().cwiseInverse())*dg_dx.value().transpose();
        const Eigen::Matrix<T, NX, NX> Lambda = dg_dx.value()*L;

        G.block(0, 0, NX, NX).noalias() +=  Lambda;
      }
      
      Eigen::Matrix<T, NX+NH, 1> descent_direction = G.fullPivLu().solve(b);

      if (NG) {
        Eigen::Matrix<T, NX, 1> del_x = descent_direction.block(0, 0, NX, 1);
        Eigen::Matrix<T, NG, 1> del_lambda = -1*L*del_x;

        for (size_t idx = 0; idx < NG; idx++) {
          if (del_lambda[idx] < 0){
            T temp_alpha = -combined_candidate[idx]/del_lambda[idx];

            if (0 <= temp_alpha)
              alpha = std::min(alpha, temp_alpha);

            }
        }
      }

      Eigen::Matrix<T, NX+NH, 1> solution =(combined_candidate + alpha*descent_direction).eval();

      return solution;
    };

private:
    double m_mu;
};

#endif // IPOPT_H
