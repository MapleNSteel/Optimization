#ifndef NLP_H
#define NLP_H

#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Dense>

/**
 * @brief A generic class that describes a Nonlinear-Program with 'NG' inequality and 'NH' equality constraints.
 * 
 * @tparam T The type used for the optimization
 * @tparam NX Size of the optimization vector
 * @tparam NG Number of inequality constraints
 * @tparam NH Number of equality constraints
 */
template <typename T, size_t NX, size_t NG=0, size_t NH=0>
class NLP {
public:
    NLP() = default;
    virtual ~NLP() = default;
    
    /**
     * @brief Pure virtual function for the objective function.
     * 
     * @param candidate_vector The initial solution vector.
     */
    virtual const T objectiveFunction(const Eigen::Matrix<T, NX, 1>& candidate_vector) const = 0;
    
    /**
     * @brief Virtual function for the gradient function.
     * 
     * @param candidate_vector The initial solution vector.
     */
    virtual const Eigen::Matrix<T, NX, 1> gradientFunction(const Eigen::Matrix<T, NX, 1>& candidate_vector) const = 0;

    /**
     * @brief Virtual function for the hessian function.
     * 
     * @param candidate_vector The initial solution vector.
     */
    virtual const Eigen::Matrix<T, NX, NX> hessianFunction(const Eigen::Matrix<T, NX, 1>& candidate_vector) const = 0; 

    /**
     * @brief Virtual function for the inequality constraint vector
     * 
     * @param candidate_vector The initial solution vector.
     */ 
    virtual const std::optional<Eigen::Matrix<T, NG, 1>> inequalityConstraintVector(const Eigen::Matrix<T, NX, 1>& candidate_vector) const {
        return std::nullopt;
    }

    /**
     * @brief Virtual function for the inequality constraint vector
     * 
     * @param candidate_vector The initial solution vector.
     */ 
    virtual const std::optional<Eigen::Matrix<T, NH, 1>> equalityConstraintVector(const Eigen::Matrix<T, NX, 1>& candidate_vector) const {
        return std::nullopt;
    }

    /**
     * @brief Virtual function for the inequality constraint vector
     * 
     * @param candidate_vector The initial solution vector.
     */ 
    virtual const std::optional<Eigen::Matrix<T, NX, NG>> gradientInequalityConstraintVector(const Eigen::Matrix<T, NX, 1>& candidate_vector) const {
        return std::nullopt;
    }

    /**
     * @brief Virtual function for the inequality constraint vector
     * 
     * @param candidate_vector The initial solution vector.
     */ 
    virtual const std::optional<Eigen::Matrix<T, NX, NH>> gradientEqualityConstraintVector(const Eigen::Matrix<T, NX, 1>& candidate_vector) const {
        return std::nullopt;
    }
};

#endif // NLP_H
