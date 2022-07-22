/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/costs.hpp"

#include "proxddp/modelling/quad-costs.hpp"
#include "proxddp/modelling/composite-costs.hpp"
#include "proxddp/modelling/sum-of-costs.hpp"

namespace proxddp {
namespace python {

void exposeCosts() {
  using context::Scalar;
  using context::CostBase;
  using context::CostData;
  using context::MatrixXs;
  using context::StageFunction;
  using context::VectorXs;

  bp::register_ptr_to_python<shared_ptr<CostBase>>();

  bp::class_<internal::PyCostFunction<>, boost::noncopyable>(
      "CostAbstract", "Base class for cost functions.",
      bp::init<const int, const int>(bp::args("self", "ndx", "nu")))
      .def("evaluate", bp::pure_virtual(&CostBase::evaluate),
           bp::args("self", "x", "u", "data"), "Evaluate the cost function.")
      .def("computeGradients", bp::pure_virtual(&CostBase::evaluate),
           bp::args("self", "x", "u", "data"),
           "Compute the cost function gradients.")
      .def("computeHessians",
           bp::pure_virtual(&CostBase::computeHessians),
           bp::args("self", "x", "u", "data"),
           "Compute the cost function hessians.")
      .add_property("ndx", &CostBase::ndx)
      .add_property("nu", &CostBase::nu)
      .def(CreateDataPythonVisitor<CostBase>());

  bp::register_ptr_to_python<shared_ptr<CostData>>();
  bp::class_<CostData>(
      "CostData", "Cost function data struct.",
      bp::init<const int, const int>(bp::args("self", "ndx", "nu")))
      .def_readwrite("value", &CostData::value_)
      .add_property(
          "Lx", bp::make_getter(&CostData::Lx_,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property(
          "Lu", bp::make_getter(&CostData::Lu_,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property("Lxx", bp::make_getter(
                               &CostData::Lxx_,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Lxu", bp::make_getter(
                               &CostData::Lxu_,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Lux", bp::make_getter(
                               &CostData::Lux_,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Luu", bp::make_getter(
                               &CostData::Luu_,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("grad", bp::make_getter(
                                &CostData::grad_,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property(
          "hess",
          bp::make_getter(&CostData::hess_,
                          bp::return_value_policy<bp::return_by_value>()));

  pinpy::StdVectorPythonVisitor<std::vector<shared_ptr<CostBase>>,
                                true>::expose("StdVec_CostAbstract",
                                              "Vector of cost objects.");
  pinpy::StdVectorPythonVisitor<std::vector<shared_ptr<CostData>>,
                                true>::expose("StdVec_CostData",
                                              "Vector of CostData objects.");

  bp::class_<ConstantCostTpl<Scalar>, bp::bases<CostBase>>(
    "ConstantCost", "A constant cost term.",
    bp::init<int, int, Scalar>(bp::args("self", "ndx", "nu", "value")))
    .def_readwrite("value", &ConstantCostTpl<Scalar>::value_)
    .def(CopyableVisitor<QuadraticCostTpl<Scalar>>());

  bp::class_<QuadraticCostTpl<Scalar>, bp::bases<CostBase>>(
      "QuadraticCost", "Quadratic cost in both state and control.",
      bp::init<const MatrixXs &, const MatrixXs &, const VectorXs &,
               const VectorXs &>(
          bp::args("self", "w_x", "w_u", "interp_x", "interp_u")))
      .def(bp::init<const MatrixXs &, const MatrixXs &>(
          "Constructor with just weights.", bp::args("self", "w_x", "w_u")))
      .def(CopyableVisitor<QuadraticCostTpl<Scalar>>());

  /* Composite costs */

  using CompositeData = CompositeCostDataTpl<Scalar>;
  using QuadResCost = QuadraticResidualCostTpl<Scalar>;
  bp::register_ptr_to_python<shared_ptr<QuadResCost>>();

  bp::class_<QuadResCost, bp::bases<CostBase>>(
      "QuadraticResidualCost", "Weighted 2-norm of a given residual function.",
      bp::init<const shared_ptr<StageFunction> &, const context::MatrixXs &>(
          bp::args("self", "function", "weights")))
      .def_readwrite("residual", &QuadResCost::residual_)
      .def_readwrite("weights", &QuadResCost::weights_)
      .def(CopyableVisitor<QuadResCost>());

  bp::register_ptr_to_python<shared_ptr<CompositeData>>();
  bp::class_<CompositeData, bp::bases<CostData>>("CompositeCostData",
                                                  bp::init<int, int>())
      .def_readwrite("residual_data", &CompositeData::residual_data);

  /* Cost stack */

  using CostStack = CostStackTpl<Scalar>;
  using CostStackData = CostStackDataTpl<Scalar>;

  bp::class_<CostStack, bp::bases<CostBase>>(
      "CostStack", "A weighted sum of other cost functions.",
      bp::init<const int, const int,
              const std::vector<shared_ptr<CostBase>> &,
              const std::vector<Scalar> &>((
          bp::arg("self"), bp::arg("ndx"), bp::arg("nu"),
          bp::arg("components") = bp::list(), bp::arg("weights") = bp::list())))
      .def_readwrite("components", &CostStack::components_,
                     "Components of this cost stack.")
      .def_readonly("weights", &CostStack::weights_,
                    "Weights of this cost stack.")
      .def("addCost", &CostStack::addCost,
          (bp::arg("self"), bp::arg("cost"), bp::arg("weight") = 1.),
          "Add a cost to the stack of costs.")
      .def("size", &CostStack::size, "Get the number of cost components.")
      .def(CopyableVisitor<CostStack>());

  bp::register_ptr_to_python<shared_ptr<CostStackData>>();
  bp::class_<CostStackData, bp::bases<CostData>>(
      "CostStackData", "Data struct for CostStack.", bp::no_init)
      .add_property(
          "sub_cost_data",
          bp::make_getter(&CostStackData::sub_cost_data,
                          bp::return_value_policy<bp::return_by_value>()));
}

} // namespace python
} // namespace proxddp
