/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./barrier-residual-cost.hpp"
#include <iostream>

namespace aligator {

template <typename Scalar>
BarrierResidualCostTpl<Scalar>::BarrierResidualCostTpl(
    xyz::polymorphic<Manifold> space, xyz::polymorphic<StageFunction> function,
    const double alpha)
    : Base(space, function->nu), alpha_(alpha), residual_(function) {}

template <typename Scalar>
void BarrierResidualCostTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                              const ConstVectorRef &u,
                                              CostData &data_) const {
  Data &data = static_cast<Data &>(data_);
  StageFunctionDataTpl<Scalar> &under_data = *data.residual_data;
  residual_->evaluate(x, u, x, under_data);
  ALIGATOR_NOMALLOC_SCOPED;
  double d = under_data.value_.norm();
  if (d < alpha_) {
    data.value_ = Scalar(0.5) * (d - alpha_) * (d - alpha_);
  } else {
    data.value_ = Scalar(0.0);
  }
}

template <typename Scalar>
void BarrierResidualCostTpl<Scalar>::computeGradients(const ConstVectorRef &x,
                                                      const ConstVectorRef &u,
                                                      CostData &data_) const {
  Data &data = static_cast<Data &>(data_);
  StageFunctionDataTpl<Scalar> &under_data = *data.residual_data;
  residual_->computeJacobians(x, u, x, under_data);
  double d = under_data.value_.norm();
  const Eigen::Index size = data.grad_.size();
  ALIGATOR_NOMALLOC_SCOPED;
  MatrixRef J = under_data.jac_buffer_.leftCols(size);
  std::cout << "Jacobian res " << J << std::endl;
  std::cout << "d " << d << std::endl;
  std::cout << "value " << under_data.value_ << std::endl;
  if (d < alpha_) {
    data.grad_.noalias() = (d - alpha_) / d * J.transpose() * under_data.value_;
  } else {
    data.grad_.setZero();
  }
}

template <typename Scalar>
void BarrierResidualCostTpl<Scalar>::computeHessians(const ConstVectorRef &x,
                                                     const ConstVectorRef &u,
                                                     CostData &data_) const {
  ALIGATOR_NOMALLOC_SCOPED;
  Data &data = static_cast<Data &>(data_);
  StageFunctionDataTpl<Scalar> &under_data = *data.residual_data;
  double d = under_data.value_.norm();
  const Eigen::Index size = data.grad_.size();
  MatrixRef J = under_data.jac_buffer_.leftCols(size);
  if (d < alpha_) {
    data.hess_.noalias() = alpha_ * J.transpose() * under_data.value_ *
                           under_data.value_.transpose() * J / std::pow(d, 3);
    data.hess_ += (d - alpha_) / d * J.transpose() * J;
    if (!gauss_newton) {
      ALIGATOR_NOMALLOC_END;
      data.Wv_buf.noalias() = (d - alpha_) / d * under_data.value_;
      residual_->computeVectorHessianProducts(x, u, x, data.Wv_buf, under_data);
      data.hess_ += under_data.vhp_buffer_;
    }
  } else {
    data.hess_.setZero();
  }
}

} // namespace aligator
