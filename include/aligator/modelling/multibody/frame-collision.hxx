#pragma once

#include "aligator/modelling/multibody/frame-collision.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace aligator {

template <typename Scalar>
void FrameCollisionResidualTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                 BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;
  pinocchio::forwardKinematics(model, pdata, x.head(model.nq));
  pinocchio::updateFramePlacements(model, pdata);

  // computes the collision distance between pair of frames
  pinocchio::updateGeometryPlacements(model, pdata, *geom_model_.get(),
                                      d.geometry_, x.head(model.nq));
  pinocchio::computeDistance(*geom_model_.get(), d.geometry_, frame_pair_id_);

  // calculate residual
  d.value_ = d.geometry_.distanceResults[frame_pair_id_].nearest_points[0] -
             d.geometry_.distanceResults[frame_pair_id_].nearest_points[1];
}

template <typename Scalar>
void FrameCollisionResidualTpl<Scalar>::computeJacobians(const ConstVectorRef &,
                                                         BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  const Model &model = *pin_model_;
  pinocchio::DataTpl<Scalar> &pdata = d.pin_data_;

  // calculate vector from joint to collision p1, expressed
  // in local world aligned
  d.distance_ = d.geometry_.distanceResults[frame_pair_id_].nearest_points[0] -
                pdata.oMi[joint_id_].translation();
  pinocchio::computeJointJacobians(model, pdata);
  pinocchio::getJointJacobian(model, pdata, joint_id_,
                              pinocchio::LOCAL_WORLD_ALIGNED, d.Jcol_);

  // compute Jacobian at p1
  d.Jcol_.template topRows<3>().noalias() +=
      pinocchio::skew(d.distance_).transpose() *
      d.Jcol_.template bottomRows<3>();

  // compute the residual derivatives
  d.Jx_.topLeftCorner(3, model.nv) = d.Jcol_.template topRows<3>();
}

template <typename Scalar>
FrameCollisionDataTpl<Scalar>::FrameCollisionDataTpl(
    const FrameCollisionResidualTpl<Scalar> &model)
    : Base(model.ndx1, model.nu, model.ndx2, 3), pin_data_(*model.pin_model_),
      geometry_(pinocchio::GeometryData(*model.geom_model_)),
      Jcol_(6, model.pin_model_->nv) {
  Jcol_.setZero();
}

} // namespace aligator
