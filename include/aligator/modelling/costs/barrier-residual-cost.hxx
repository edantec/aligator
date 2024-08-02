/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./barrier-residual-cost.hpp"
#include <iostream>
namespace aligator {

template <typename Scalar>
BarrierResidualCostTpl<Scalar>::BarrierResidualCostTpl(
    xyz::polymorphic<Manifold> space, xyz::polymorphic<StageFunction> function,
    const double alpha, const double weight)
    : Base(space, function->nu), alpha_(alpha), weight_(weight),
      residual_(function) {}

template <typename Scalar>
void BarrierResidualCostTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                              const ConstVectorRef &u,
                                              CostData &data_) const {
  Data &data = static_cast<Data &>(data_);
  StageFunctionDataTpl<Scalar> &under_data = *data.residual_data;
  residual_->evaluate(x, u, x, under_data);
  ALIGATOR_NOMALLOC_SCOPED;
  // data.value_ = .5 * under_data.value_.dot(under_data.value_);
  double d = under_data.value_.norm();
  if (d < alpha_) {
    data.value_ = Scalar(0.5) * weight_ * (d - alpha_) * (d - alpha_);
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

  const Eigen::Index size = data.grad_.size();
  ALIGATOR_NOMALLOC_SCOPED;
  MatrixRef J = under_data.jac_buffer_.leftCols(size);
  double d = under_data.value_.norm();
  // data.grad_.noalias() = J.transpose() * under_data.value_;
  /* std::cout << "grad " << data.grad_.size() << std::endl;
  std::cout << " uv" <<  under_data.value_.size() << std::endl;
  std::cout << "J size " << J.rows() << ", " << J.cols() << std::endl;*/
  if (d < alpha_ and d > 0) {
    data.grad_.noalias() =
        weight_ * (d - alpha_) / d * J.transpose() * under_data.value_;
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

  const Eigen::Index size = data.grad_.size();
  MatrixRef J = under_data.jac_buffer_.leftCols(size);
  double d = under_data.value_.norm();
  // data.hess_.noalias() = J.transpose() * J;
  if (d < alpha_ and d > 0) {
    data.hess_.noalias() = weight_ * alpha_ * J.transpose() *
                           under_data.value_ * under_data.value_.transpose() *
                           J / std::pow(d, 3);
    data.hess_ += weight_ * (d - alpha_) / d * J.transpose() * J;
    if (!gauss_newton) {
      ALIGATOR_NOMALLOC_END;
      data.Wv_buf.noalias() = (d - alpha_) / d * under_data.value_;
      residual_->computeVectorHessianProducts(x, u, x, data.Wv_buf, under_data);
      data.hess_ += weight_ * under_data.vhp_buffer_;
    }
  } else {
    data.hess_.setZero();
  }
}

} // namespace aligator
