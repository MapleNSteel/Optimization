#ifndef SOLVER_H
#define SOLVER_H

#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "NLP/NLP.hpp"

/**
 * @brief A generic Solver class that takes two functions and solves the function using an iterative algorithm.
 * 
 * @tparam T The type used for the optimization
 */
template <typename T, size_t NX, size_t NG=0, size_t NH=0>
class Solver {
public:
  /**
   * @brief Construct a new Solver object.
   * 
   * @param nlp Underlying NLP to solved
   * @param tolerance Tolerance for convergence.
   * @param max_iterations Maximum number of iterations.
   */
  Solver(const NLP<T, NX, NG, NH>& nlp,
      T tolerance,
      size_t max_iterations) :
      m_nlp(nlp),
      m_tolerance(tolerance),
      m_max_iterations(max_iterations)
  {}

  /**
   * @brief Virtual function to solve the objective function with inequality constraints.
   * 
   * @param candidate_vector The initial solution vector.
   * @param sigma_vector The sigma vector holding co-states for inequalities
   * @param lambda_vector The sigma vector holding co-states for equalities
   * @return Eigen::Matrix<T, 2, 1> The solution vector.
   */
  virtual const Eigen::Matrix<T, NX+NG, 1> solve(const Eigen::Matrix<T, NX, 1>& candidate_vector, const std::optional<Eigen::Matrix<T, NG, 1>>& sigma_vector = std::nullopt, const std::optional<Eigen::Matrix<T, NH, 1>>& lambda_vector = std::nullopt) const = 0;
protected:
  T m_tolerance = 1e-5; ///< Tolerance for convergence.
  size_t m_max_iterations = 1000; ///< Maximum number of iterations.
  const NLP<T, NX, NG, NH>& m_nlp;
};

#endif // SOLVER_H