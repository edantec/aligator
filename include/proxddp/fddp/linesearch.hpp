/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/linesearch.hpp"

namespace proxddp {

/// @brief    The backtracking linesearch from FDDP (Mastalli et al).
/// @details  The conditions that are checked for are not exactly the Goldstein
/// conditions.
template <typename Scalar> struct FDDPGoldsteinLinesearch {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  template <typename F, typename M>
  inline static std::pair<Scalar, Scalar>
  run(const F &phi, const M &model, Scalar phi0,
      typename Linesearch<Scalar>::Options &ls_params, Scalar d1,
      Scalar th_grad) {
    Scalar th_accept_step_ = 0.1;
    Scalar th_accept_neg_step_ = 2.0;
    Scalar beta = ls_params.contraction_min;
    Scalar atry = 1.;
    Scalar phitry = phi0;
    Scalar dVreal = 0.;

    // backtrack until going under alpha_min
    do {
      try {
        phitry = phi(atry);
        dVreal = phitry - phi0;
      } catch (const ::proxddp::RuntimeError &) {
        atry *= beta;
        continue;
      }
      Scalar dVmodel = model(atry) - phi0;
      // check if descent direction
      if (dVmodel <= 0.) {
        if (-d1 < th_grad || dVreal < th_accept_step_ * dVmodel)
          break;
      } else {
        // accept a small increase in cost
        if (dVreal <= th_accept_neg_step_ * dVmodel)
          break;
      }
      atry *= beta;
    } while (atry >= ls_params.alpha_min);
    if (atry < ls_params.alpha_min) {
      atry = ls_params.alpha_min;
      phitry = phi(atry);
    }
    return std::make_pair(atry, phitry);
  }
};

} // namespace proxddp
