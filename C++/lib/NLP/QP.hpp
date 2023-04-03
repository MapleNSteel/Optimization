#ifndef QP_H
#define QP_H

#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "NLP.hpp"

template <typename T, size_t NX, size_t NG=0, size_t NH=0>
class QP : public NLP<T, NX, NG, NH> {
public:
    QP(const Eigen::Matrix<T, NX, NX>& Q,
       const Eigen::Matrix<T, NX, 1>& g,
       const std::optional<Eigen::Matrix<T, NG, NX>> G = std::nullopt,
       const std::optional<Eigen::Matrix<T, NG, 1>> h = std::nullopt,
       const std::optional<Eigen::Matrix<T, NH, NX>> A = std::nullopt,
       const std::optional<Eigen::Matrix<T, NH, 1>> b = std::nullopt)
    :  m_Q(std::move(Q)),
       m_g(std::move(g)),
       m_G(std::move(G)),
       m_h(std::move(h)),
       m_A(std::move(A)),
       m_b(std::move(b))
    {}

    virtual const T objectiveFunction(const Eigen::Matrix<T, NX, 1>& candidate_vector) const override {
        return (0.5 * candidate_vector.transpose()*m_Q*candidate_vector + m_g.transpose()*candidate_vector).value();
    }
    virtual const Eigen::Matrix<T, NX, 1> gradientFunction(const Eigen::Matrix<T, NX, 1>& candidate_vector) const override {
        return 0.5*(m_Q+m_Q.transpose())*candidate_vector + m_g;
    }
    virtual const Eigen::Matrix<T, NX, NX> hessianFunction(const Eigen::Matrix<T, NX, 1>& candidate_vector) const override {
        return 0.5*(m_Q+m_Q.transpose());
    }
protected:
    const Eigen::Matrix<T, NX, NX> m_Q;
    const Eigen::Matrix<T, NX, 1> m_g;
    const std::optional<Eigen::Matrix<T, NG, NX>> m_G;
    const std::optional<Eigen::Matrix<T, NG, 1>> m_h;
    const std::optional<Eigen::Matrix<T, NH, NX>> m_A;
    const std::optional<Eigen::Matrix<T, NH, 1>> m_b;                                                                                                                                                                                                                                
};

#endif // QP_H
