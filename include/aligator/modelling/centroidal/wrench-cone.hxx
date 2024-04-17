#pragma once

#include "aligator/modelling/centroidal/wrench-cone.hpp"

namespace aligator {

template <typename Scalar>
WrenchConeResidualTpl<Scalar>::WrenchConeResidualTpl(
    const int ndx, const int nu, const int k, const double mu,
    const double half_length, const double half_width, const Matrix3s &R)
    : Base(ndx, nu, 17), k_(k), mu_(mu), hL_(half_length), hW_(half_width) {
  A_.setZero();
  updateWrenchCone(R);
}

template <typename Scalar>
void WrenchConeResidualTpl<Scalar>::evaluate(const ConstVectorRef &,
                                             const ConstVectorRef &u,
                                             const ConstVectorRef &,
                                             BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.value_ = A_ * u.template segment<6>(k_ * 6);
}

template <typename Scalar>
void WrenchConeResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                     const ConstVectorRef &,
                                                     const ConstVectorRef &,
                                                     BaseData &data) const {
  Data &d = static_cast<Data &>(data);

  d.Ju_.template block<17, 6>(0, k_ * 6) = A_;
}

template <typename Scalar>
void WrenchConeResidualTpl<Scalar>::updateWrenchCone(const Matrix3s &R) {
  // Unilateral contact and friction cone
  // [ 0  0  -1  0  0  0
  //  -1  0 -mu  0  0  0
  //   1  0 -mu  0  0  0
  //   0 -1 -mu  0  0  0
  //   0  1 -mu  0  0  0]
  A_.row(0).head(3) = -R.col(2).transpose();
  A_.row(1).head(3) = Vector3s(-1, 0, -mu_).transpose() * R.transpose();
  A_.row(2).head(3) = Vector3s(1, 0, -mu_).transpose() * R.transpose();
  A_.row(3).head(3) = Vector3s(0, -1, -mu_).transpose() * R.transpose();
  A_.row(4).head(3) = Vector3s(0, 1, -mu_).transpose() * R.transpose();

  // Local CoP inequalities
  // [0  0 -W -1  0  0
  //  0  0 -W  1  0  0
  //  0  0 -L  0 -1  0
  //  0  0 -L  0  1  0]
  A_.row(5) << -hW_ * R.col(2).transpose(), -R.col(0).transpose();
  A_.row(6) << -hW_ * R.col(2).transpose(), R.col(0).transpose();
  A_.row(7) << -hL_ * R.col(2).transpose(), -R.col(1).transpose();
  A_.row(8) << -hL_ * R.col(2).transpose(), R.col(1).transpose();

  // z-torque limits
  // [ W  L -mu*(L+W) -mu -mu -1
  //   W -L -mu*(L+W) -mu  mu -1
  //  -W  L -mu*(L+W)  mu -mu -1
  //  -W -L -mu*(L+W)  mu  mu -1]
  Scalar mu_LW = -(hL_ + hW_) * mu_;
  A_.row(9) << Vector3s(hW_, hL_, mu_LW).transpose() * R.transpose(),
      Vector3s(-mu_, -mu_, Scalar(-1.)).transpose() * R.transpose();
  A_.row(10) << Vector3s(hW_, -hL_, mu_LW).transpose() * R.transpose(),
      Vector3s(-mu_, mu_, Scalar(-1.)).transpose() * R.transpose();
  A_.row(11) << Vector3s(-hW_, hL_, mu_LW).transpose() * R.transpose(),
      Vector3s(mu_, -mu_, Scalar(-1.)).transpose() * R.transpose();
  A_.row(12) << Vector3s(-hW_, -hL_, mu_LW).transpose() * R.transpose(),
      Vector3s(mu_, mu_, Scalar(-1.)).transpose() * R.transpose();

  // z-torque limits
  // [ W  L -mu*(L+W)  mu  mu 1
  //   W -L -mu*(L+W)  mu -mu 1
  //  -W  L -mu*(L+W) -mu  mu 1
  //  -W -L -mu*(L+W) -mu -mu 1]
  A_.row(13) << Vector3s(hW_, hL_, mu_LW).transpose() * R.transpose(),
      Vector3s(mu_, mu_, Scalar(1.)).transpose() * R.transpose();
  A_.row(14) << Vector3s(hW_, -hL_, mu_LW).transpose() * R.transpose(),
      Vector3s(mu_, -mu_, Scalar(1.)).transpose() * R.transpose();
  A_.row(15) << Vector3s(-hW_, hL_, mu_LW).transpose() * R.transpose(),
      Vector3s(-mu_, mu_, Scalar(1.)).transpose() * R.transpose();
  A_.row(16) << Vector3s(-hW_, -hL_, mu_LW).transpose() * R.transpose(),
      Vector3s(-mu_, -mu_, Scalar(1.)).transpose() * R.transpose();
}

template <typename Scalar>
WrenchConeDataTpl<Scalar>::WrenchConeDataTpl(
    const WrenchConeResidualTpl<Scalar> *model)
    : Base(model->ndx1, model->nu, model->ndx2, 17) {}

} // namespace aligator
