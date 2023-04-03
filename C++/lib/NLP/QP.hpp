#ifndef QP_H
#define QP_H

#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Dense>

template <typename T, size_t NX, size_t NG=0, size_t NH=0>
class QP : public NLP<T, NX, NG, NH> {
public:
    QP() = default;

protected:
    virtual T m_objectiveFunction(const Eigen::Matrix<T, NX, 1>& candidate_vector) override {

    }
    virtual Eigen::Matrix<T, NX, 1> gradientFunction(const Eigen::Matrix<T, NX, 1>&) override {

    }
    virtual Eigen::Matrix<T, NX, NX> hessianFunction(const Eigen::Matrix<T, NX, 1>&) override {

    }

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
private:
    Eigen::Matrix<T, NX, NX> m_Q;
    Eigen::Matrix<T, 1, NX> m_g;
    Eigen::Matrix<T, NG, NX> m_G;
    Eigen::Matrix<T, NG, 1> m_h;
    Eigen::Matrix<T, NH, NX> m_A;
    Eigen::Matrix<T, NH, 1> m_b;                                                                                                                                                                                                                                
};

#endif // QP_H
