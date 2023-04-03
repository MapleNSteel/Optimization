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

protected:
    virtual T m_objectiveFunction(const Eigen::Matrix<T, NX, 1>& candidate_vector) = 0;
    virtual Eigen::Matrix<T, NX, 1> gradientFunction(const Eigen::Matrix<T, NX, 1>&) = 0;
    virtual Eigen::Matrix<T, NX, NX> hessianFunction(const Eigen::Matrix<T, NX, 1>&) = 0;

    // Constraints
    std::optional<Eigen::Matrix<T, NG, 1>> inequalityConstraintVector(const Eigen::Matrix<T, NX, 1>&) {
        return std::nullopt;
    }
    std::optional<Eigen::Matrix<T, NH, 1>> equalityConstraintVector(const Eigen::Matrix<T, NX, 1>&) {
        return std::nullopt;
    }
    std::optional<Eigen::Matrix<T, NX, NG>> gradientInequalityConstraintVector(const Eigen::Matrix<T, NX, 1>&) {
        return std::nullopt;
    }
    std::optional<Eigen::Matrix<T, NX, NH>> gradientEqualityConstraintVector(const Eigen::Matrix<T, NX, 1>&) {
        return std::nullopt;
    }
};

#endif // NLP_H
