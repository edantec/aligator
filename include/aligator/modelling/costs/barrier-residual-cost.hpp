/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./composite-costs.hpp"

namespace aligator {

/** @brief Quadratic barrier composite of an underlying function.
 *
 * @details This is defined as
 * \f[
 *      c(x) \overset{\triangle}{=} \frac{1}{2} \(|r(x)\| - \alpha \)^2 if |r| <
 \alpha, 0 otherwise
 * \f]
 */

template <typename _Scalar>
struct BarrierResidualCostTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using Data = CompositeCostDataTpl<Scalar>;
  using StageFunction = StageFunctionTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  double alpha_;
  xyz::polymorphic<StageFunction> residual_;
  bool gauss_newton = true;

  BarrierResidualCostTpl(xyz::polymorphic<Manifold> space,
                         xyz::polymorphic<StageFunction> function,
                         const double alpha);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostData &data_) const;

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data_) const;

  void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                       CostData &data_) const;

  shared_ptr<CostData> createData() const {
    return std::make_shared<Data>(this->ndx(), this->nu,
                                  residual_->createData());
  }
};

extern template struct BarrierResidualCostTpl<context::Scalar>;

} // namespace aligator

#include "./barrier-residual-cost.hxx"
