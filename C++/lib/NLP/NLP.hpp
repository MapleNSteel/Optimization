#ifndef NLP_H
#define NLP_H

#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Dense>

template <typename T, size_t NX, size_t NG=0, size_t NH=0>
class NLP {
public:
    NLP() = default;
    virtual ~NLP() = default;
    
    virtual const T objectiveFunction(const Eigen::Matrix<T, NX, 1>& candidate_vector) const = 0;
    virtual const Eigen::Matrix<T, NX, 1> gradientFunction(const Eigen::Matrix<T, NX, 1>& candidate_vector) const = 0;
    virtual const Eigen::Matrix<T, NX, NX> hessianFunction(const Eigen::Matrix<T, NX, 1>& candidate_vector) const = 0;

    // Constraints
    virtual const std::optional<Eigen::Matrix<T, NG, 1>> inequalityConstraintVector(const Eigen::Matrix<T, NX, 1>& candidate_vector) const {
        return std::nullopt;
    }
    virtual const std::optional<Eigen::Matrix<T, NH, 1>> equalityConstraintVector(const Eigen::Matrix<T, NX, 1>& candidate_vector) const {
        return std::nullopt;
    }
    virtual const std::optional<Eigen::Matrix<T, NX, NG>> gradientInequalityConstraintVector(const Eigen::Matrix<T, NX, 1>& candidate_vector) const {
        return std::nullopt;
    }
    virtual const std::optional<Eigen::Matrix<T, NX, NH>> gradientEqualityConstraintVector(const Eigen::Matrix<T, NX, 1>& candidate_vector) const {
        return std::nullopt;
    }
};

#endif // NLP_H
