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
  Solver(NLP<T, NX, NG, NH>& nlp,
      T tolerance,
      size_t max_iterations) :
      m_nlp(nlp),
      m_tolerance(tolerance),
      m_max_iterations(max_iterations)
  {}

  /**
   * @brief Virtual function to solve the objective function using an iterative algorithm.
   * 
   * @param initial_solution_vector The initial solution vector.
   * @return Eigen::Matrix<T, 2, 1> The solution vector.
   */
  virtual Eigen::Matrix<T, NX, 1> const solve(Eigen::Matrix<T, NX, 1> initial_solution_vector) = 0;
protected:
  std::shared_ptr<NLP<T, NX, NG, NH>> m_nlp;
  T m_tolerance = 1e-5; ///< Tolerance for convergence.
  size_t m_max_iterations = 1000; ///< Maximum number of iterations.
};

#endif // SOLVER_H