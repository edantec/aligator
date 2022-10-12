#pragma once

#include "proxddp/modelling/dynamics/ode-abstract.hpp"

#include <proxnlp/modelling/spaces/multibody.hpp>
#include <pinocchio/multibody/data.hpp>

namespace proxddp {
namespace dynamics {
template <typename Scalar> struct MultibodyFreeFwdDataTpl;

/**
 * @brief   Free-space multibody forward dynamics, using Pinocchio.
 *
 * @details This is described in state-space \f$\mathcal{X} = T\mathcal{Q}\f$
 * (the phase space \f$x = (q,v)\f$) using the differential equation \f[ \dot{x}
 * = f(x, u) = \begin{bmatrix} v \\ a(q, v, \tau(u)) \end{bmatrix} \f] where
 * \f$\tau(u) = Bu\f$, \f$B\f$ is a given actuation matrix, and
 * \f$a(q,v,\tau)\f$ is the acceleration computed from the ABA algorithm.
 *
 */
template <typename _Scalar>
struct MultibodyFreeFwdDynamicsTpl : ODEAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = ODEAbstractTpl<Scalar>;
  using BaseData = ODEDataTpl<Scalar>;
  using ContDataAbstract = ContinuousDynamicsDataTpl<Scalar>;
  using Data = MultibodyFreeFwdDataTpl<Scalar>;
  using Manifold = proxnlp::MultibodyPhaseSpace<Scalar>;
  using ManifoldPtr = shared_ptr<Manifold>;

  using Base::nu_;

  ManifoldPtr space_;
  MatrixXs actuation_matrix_;
  Eigen::FullPivLU<MatrixXs> lu_decomp;

  const Manifold &space() const { return *space_; }
  int ntau() const { return space_->getModel().nv; }

  MultibodyFreeFwdDynamicsTpl(const ManifoldPtr &state,
                              const MatrixXs &actuation);
  MultibodyFreeFwdDynamicsTpl(const ManifoldPtr &state);

  /**
   * @brief  Determine whether the system is underactuated.
   * @details This is the case when the actuation matrix rank is lower to the
   * velocity dimension.
   */
  bool isUnderactuated() const {
    long nv = space().getModel().nv;
    return act_matrix_rank < nv;
  }

  Eigen::Index getActuationMatrixRank() const { return act_matrix_rank; }

  virtual void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                       BaseData &data) const;
  virtual void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                        BaseData &data) const;

  shared_ptr<ContDataAbstract> createData() const;

private:
  Eigen::Index act_matrix_rank;
};

template <typename Scalar> struct MultibodyFreeFwdDataTpl : ODEDataTpl<Scalar> {
  using Base = ODEDataTpl<Scalar>;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using PinDataType = pinocchio::DataTpl<Scalar>;
  VectorXs tau_;
  MatrixXs dtau_dx_;
  MatrixXs dtau_du_;
  shared_ptr<PinDataType> pin_data_;
  MultibodyFreeFwdDataTpl(const MultibodyFreeFwdDynamicsTpl<Scalar> *cont_dyn);
};

} // namespace dynamics
} // namespace proxddp

#include "proxddp/modelling/dynamics/multibody-free-fwd.hxx"
