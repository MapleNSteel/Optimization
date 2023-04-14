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
       const Eigen::Matrix<T, NX, 1>& g)
    :  m_Q(std::move(Q)),
       m_g(std::move(g))
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
};

#endif // QP_H
