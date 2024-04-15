/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/ode-abstract.hpp"

#include <proxsuite-nlp/modelling/spaces/multibody.hpp>
#include <proxsuite-nlp/modelling/spaces/cartesian-product.hpp>
#include <pinocchio/multibody/model.hpp>

namespace aligator {
namespace dynamics {
template <typename Scalar> struct VkinodynamicsFwdDataTpl;

/**
 * @brief   Nonlinear centroidal and full kinematics forward dynamics.
 *
 * @details This is described in state-space \f$\mathcal{X} = T\mathcal{Q}\f$
 * (the phase space \f$x = (q,v)\f$ with q kinematics pose and v joint velocity)
 * using the differential equation \f$ \dot{q} = v \f$ and
 * \f$ \dot{v} = \begin{bmatrix} a_u \\ a_j \end{bmatrix} \f$,
 * with \f$a_u\f$ base acceleration computed from centroidal Newton-Euler law of
 * momentum
 * (\f$ \dot{H} = \dot{A}\dot{q} + A\ddot{q} = \begin{bmatrix} \sum_i=1^{n_k}
 * f_i + mg \\ \sum_i=1^{n_k} (p_i - c) \times f_i \end{bmatrix} \f$ ) and
 * \f$a_j\f$ commanded joints acceleration.
 *
 */
template <typename _Scalar>
struct VkinodynamicsFwdDynamicsTpl : ODEAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ODEAbstractTpl<Scalar>;
  using BaseData = ODEDataTpl<Scalar>;
  using ContDataAbstract = ContinuousDynamicsDataTpl<Scalar>;
  using Data = VkinodynamicsFwdDataTpl<Scalar>;
  using Manifold = proxsuite::nlp::CartesianProductTpl<Scalar>;
  using ManifoldPtr = shared_ptr<Manifold>;
  using Model = pinocchio::ModelTpl<Scalar>;
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;

  using Base::nu_;

  ManifoldPtr space_;
  Model pin_model_;
  double mass_;
  Vector3s gravity_;
  int force_size_;
  int nj_;
  int nf_;

  std::vector<bool> contact_states_;
  std::vector<pinocchio::FrameIndex> contact_ids_;

  const Manifold &space() const { return *space_; }

  VkinodynamicsFwdDynamicsTpl(
      const ManifoldPtr &state, const Model &model, const Vector3s &gravity,
      const std::vector<bool> &contact_states,
      const std::vector<pinocchio::FrameIndex> &contact_ids,
      const int force_size);

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               BaseData &data) const;
  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const;

  shared_ptr<ContDataAbstract> createData() const;
};

template <typename Scalar> struct VkinodynamicsFwdDataTpl : ODEDataTpl<Scalar> {
  using Base = ODEDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using Matrix6Xs = typename math_types<Scalar>::Matrix6Xs;
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;
  using Matrix6s = Eigen::Matrix<Scalar, 6, 6>;
  using Vector6s = Eigen::Matrix<Scalar, 6, 1>;

  PinData pin_data_;
  Matrix6Xs dh_dq_;
  Matrix6Xs dhdot_dq_;
  Matrix6Xs dhdot_dv_;
  Matrix6Xs dhdot_da_;
  Matrix6Xs fJf_;
  VectorXs v0_;
  VectorXs a0_;

  Vector6s cforces_;
  Matrix3s Jtemp_;
  Matrix6s Agu_inv_;

  VkinodynamicsFwdDataTpl(const VkinodynamicsFwdDynamicsTpl<Scalar> *model);
};

} // namespace dynamics
} // namespace aligator

#include "aligator/modelling/dynamics/vkinodynamics-fwd.hxx"
