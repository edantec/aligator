/// @file    workspace.hpp
/// @brief   Define workspace for the ProxDDP solver.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/workspace-common.hpp"
#include "proxddp/core/proximal-penalty.hpp"

#include <Eigen/Cholesky>
#if PROXDDP_CUSTOM_LDLT
#include <proxnlp/blocks.hpp>
#endif

namespace proxddp {

namespace chol = proxnlp::block_chol;

/** @brief Workspace for solver SolverProxDDP.
 *
 * @details This struct holds data for the Riccati forward and backward passes,
 *          the primal-dual steps, problem data...
 */
template <typename Scalar> struct WorkspaceTpl : WorkspaceBaseTpl<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using ProxPenalty = ProximalPenaltyTpl<Scalar>;
  using ProxData = typename ProxPenalty::Data;
  using StageModel = StageModelTpl<Scalar>;
  using Base = WorkspaceBaseTpl<Scalar>;
#if PROXDDP_CUSTOM_LDLT
  using LDLT = ::proxnlp::block_chol::DenseLDLT;
#else
  using LDLT = Eigen::LDLT<MatrixXs, Eigen::Lower>;
#endif

  using Base::problem_data;
  using Base::q_params;
  using Base::trial_us;
  using Base::trial_xs;
  using Base::value_params;

  /// Proximal penalty data.
  std::vector<shared_ptr<ProxData>> prox_datas;

  /// Lagrange multipliers for ALM & linesearch.
  std::vector<VectorXs> trial_lams;
  std::vector<VectorXs> lams_plus;
  std::vector<VectorXs> lams_pdal;
  /// Shifted constraints the projection operators should be applied to.
  std::vector<VectorXs> shifted_constraints;

  /// @name Riccati gains, memory buffers for primal-dual steps
  /// @{
  std::vector<VectorXs> pd_step_;
  std::vector<VectorRef> dxs;
  std::vector<VectorRef> dus;
  std::vector<VectorRef> dlams;

  /// Buffer for KKT matrix
  std::vector<MatrixXs> kkt_mats_;
  /// Buffer for KKT right hand side
  std::vector<MatrixXs> kkt_rhs_;
  /// Linear system residual buffers: used for iterative refinement
  std::vector<MatrixXs> kkt_resdls_;
  /// LDLT decompositions
  std::vector<LDLT> ldlts_;

  /// @}

  /// @name Previous external/proximal iterates
  /// @{

  std::vector<VectorXs> prev_xs;
  std::vector<VectorXs> prev_us;
  std::vector<VectorXs> lams_prev;

  /// @}

  /// Subproblem termination criterion for each stage.
  VectorXs stage_inner_crits;
  /// Constraint violation for each stage and each constraint of the
  /// TrajOptProblemTpl.
  std::vector<VectorXs> stage_prim_infeas;
  /// Dual infeasibility for each stage of the TrajOptProblemTpl.
  VectorXs stage_dual_infeas;

  /// Overall subproblem termination criterion.
  Scalar inner_criterion = 0.;

  explicit WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem);

  void cycle_left();

  /// @brief Same as cycle_left(), but add a StageDataTpl to problem_data.
  /// @details The implementation pushes back on top of the vector of
  /// StageDataTpl, rotates left, then pops the first element back out.
  void cycle_append(const shared_ptr<StageModel> &stage);

  template <typename T>
  friend std::ostream &operator<<(std::ostream &oss,
                                  const WorkspaceTpl<T> &self);
};

} // namespace proxddp

#include "proxddp/core/workspace.hxx"
