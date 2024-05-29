/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#include <proxsuite-nlp/python/polymorphic.hpp>
#include "aligator/python/functions.hpp"

#include "aligator/modelling/function-xpr-slice.hpp"

namespace aligator {
namespace python {
using context::ConstMatrixRef;
using context::ConstVectorRef;
using context::StageFunction;
using context::StageFunctionData;
using context::UnaryFunction;
using internal::PyUnaryFunction;
using PolyUnaryFunction = xyz::polymorphic<UnaryFunction>;

/// Expose the UnaryFunction type and its member function overloads.
void exposeUnaryFunctions() {
  proxsuite::nlp::python::register_polymorphic_to_python<PolyUnaryFunction>();
  using unary_eval_t = void (UnaryFunction::*)(const ConstVectorRef &,
                                               StageFunctionData &) const;
  using full_eval_t =
      void (UnaryFunction::*)(const ConstVectorRef &, const ConstVectorRef &,
                              const ConstVectorRef &, StageFunctionData &)
          const;
  using unary_vhp_t =
      void (UnaryFunction::*)(const ConstVectorRef &, const ConstVectorRef &,
                              StageFunctionData &) const;
  using full_vhp_t = void (UnaryFunction::*)(
      const ConstVectorRef &, const ConstVectorRef &, const ConstVectorRef &,
      const ConstVectorRef &, StageFunctionData &) const;
  bp::class_<PyUnaryFunction<>, bp::bases<StageFunction>, boost::noncopyable>(
      "UnaryFunction",
      "Base class for unary functions of the form :math:`x \\mapsto f(x)`.",
      bp::no_init)
      .def(bp::init<const int, const int, const int, const int>(
          ("self"_a, "ndx1", "nu", "ndx2", "nr")))
      .def("evaluate", bp::pure_virtual<unary_eval_t>(&UnaryFunction::evaluate),
           ("self"_a, "x", "data"))
      .def<full_eval_t>("evaluate", &UnaryFunction::evaluate,
                        ("self"_a, "x", "u", "y", "data"))
      .def("computeJacobians",
           bp::pure_virtual<unary_eval_t>(&UnaryFunction::computeJacobians),
           ("self"_a, "x", "data"))
      .def<full_eval_t>("computeJacobians", &UnaryFunction::computeJacobians,
                        ("self"_a, "x", "u", "y", "data"))
      .def<unary_vhp_t>(
          "computeVectorHessianProducts",
          &UnaryFunction::computeVectorHessianProducts,
          &PyUnaryFunction<>::default_computeVectorHessianProducts,
          ("self"_a, "x", "lbda", "data"))
      .def<full_vhp_t>("computeVectorHessianProducts",
                       &UnaryFunction::computeVectorHessianProducts,
                       ("self"_a, "x", "u", "y", "lbda", "data"));
  //.def(SlicingVisitor<UnaryFunction>());
}

} // namespace python
} // namespace aligator
