/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/modelling/dynamics/vkinodynamics-fwd.hpp"

namespace aligator {
namespace dynamics {

template struct VkinodynamicsFwdDynamicsTpl<context::Scalar>;
template struct VkinodynamicsFwdDataTpl<context::Scalar>;

} // namespace dynamics
} // namespace aligator
