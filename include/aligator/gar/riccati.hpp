/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once
#include "aligator/context.hpp"
#include "./lqr-knot.hpp"
#include "./BlkMatrix.hpp"

#include <Eigen/Cholesky>

namespace aligator {
namespace gar {

/// A sequential, regularized Riccati algorithm
// for proximal-regularized, constrained LQ problems.
template <typename Scalar> class ProximalRiccatiSolver {
public:
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;
  using knot_t = LQRKnot<Scalar>;
  using vecvec_t = std::vector<VectorXs>;

  struct value_t {
    MatrixXs Pmat;  //< Riccati matrix
    VectorXs pvec;  //< Riccati bias
    MatrixXs Lbmat; //< Dual-space system matrix
    MatrixXs Vmat;  //< "cost-to-go" matrix
    VectorXs vvec;
    Eigen::LLT<MatrixXs> chol;
    value_t(uint nx)
        : Pmat(nx, nx), pvec(nx), Lbmat(nx, nx), //
          Vmat(nx, nx), vvec(nx), chol(nx) {}
  };

  struct kkt_t {
    BlkMatrix<MatrixXs, 2, 2> matrix;
    Eigen::LDLT<MatrixXs> chol;
    kkt_t(uint nu, uint nc) : matrix({nu, nc}), chol(matrix.rows()) {
      matrix.setZero();
    }
    MatrixRef R() { return matrix(0, 0); };
    MatrixRef D() { return matrix(1, 0); };
    auto dual() { return matrix(1, 1).diagonal(); }
  };

  struct hmlt_t {
    MatrixXs Qhat, Rhat, Shat;
    VectorXs qhat, rhat;
    RowMatrixXs AtV;
    RowMatrixXs BtV;
    hmlt_t(uint nx, uint nu)
        : Qhat(nx, nx), Rhat(nu, nu), Shat(nx, nu), //
          qhat(nx), rhat(nu), AtV(nx, nx), BtV(nu, nx) {}
  };
  struct error_t {
    Scalar lbda;
    Scalar pm;
    Scalar pv;
    Scalar fferr;
    Scalar fberr;
  };

  /// Per-node struct for all computations in the factorization.
  struct stage_solve_data_t {
    stage_solve_data_t(uint nx, uint nu, uint nc)
        : ff({nu, nc}, {1}), fb({nu, nc}, {nx}), ffRhs(nu + nc),
          fbRhs(nu + nc, nx), kkt(nu, nc), hmlt(nx, nu), vm(nx), PinvEt(nx, nx),
          wvec(nx) {
      ff.setZero();
      fb.setZero();
    }

    BlkMatrix<VectorXs, 2, 1> ff; //< feedforward gain
    BlkMatrix<MatrixXs, 2, 1> fb; //< feedback gain
    VectorXs ffRhs;
    MatrixXs fbRhs;
    kkt_t kkt;       //< KKT matrix buffer
    hmlt_t hmlt;     //< stage system data
    value_t vm;      //< cost-to-go parameters
    MatrixXs PinvEt; //< tmp buffer for \f$EP^{-1}\f$
    VectorXs wvec;   //< tmp buffer for \f$-P^{-1}p\f$
    error_t err;     //< numerical errors
  };

  explicit ProximalRiccatiSolver(const LQRProblem<Scalar> &problem)
      : datas(), kkt0(problem.stages[0].nx, (uint)problem.nc0()),
        problem(problem) {
    initialize();
  }

  ProximalRiccatiSolver(LQRProblem<Scalar> &&problem) = delete;

  void computeKktTerms(const knot_t &model, stage_solve_data_t &d,
                       const value_t &vnext);

  /// Backward sweep.
  bool backward(Scalar mudyn, Scalar mueq);
  /// Forward sweep.
  bool forward(vecvec_t &xs, vecvec_t &us, vecvec_t &vs, vecvec_t &lbdas) const;

  std::vector<stage_solve_data_t> datas;
  struct kkt0_t {
    BlkMatrix<MatrixXs, 2, 2> mat;
    BlkMatrix<VectorXs, 2, 1> rhs{mat.rowDims()};
    Eigen::LDLT<MatrixXs> chol{mat.rows()};
    kkt0_t(uint nx, uint nc) : mat({nx, nc}) {}
  } kkt0;

protected:
  void initialize() {
    auto N = size_t(problem.horizon());
    auto &knots = problem.stages;
    datas.reserve(N + 1);
    for (size_t t = 0; t <= N; t++) {
      const knot_t &knot = knots[t];
      datas.emplace_back(knot.nx, knot.nu, knot.nc);
    }
  }

private:
  const LQRProblem<Scalar> &problem;
};

} // namespace gar
} // namespace aligator

#include "./riccati.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./riccati.txx"
#endif