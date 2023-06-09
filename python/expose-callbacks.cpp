/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/core/callback-base.hpp"
#include "proxddp/helpers/history-callback.hpp"
#include "proxddp/helpers/linesearch-callback.hpp"

namespace proxddp {
namespace python {

using context::Scalar;
using CallbackBase = helpers::base_callback<Scalar>;

struct CallbackWrapper : CallbackBase, bp::wrapper<CallbackBase> {
  CallbackWrapper() = default;
  void call(const WorkspaceBaseTpl<context::Scalar> &w,
            const ResultsBaseTpl<context::Scalar> &r) {
    PROXDDP_PYTHON_OVERRIDE_PURE(void, "call", boost::cref(w), boost::cref(r));
  }
};

void exposeHistoryCallback() {
  using history_callback_t = helpers::history_callback<Scalar>;
  using history_storage_t = decltype(history_callback_t::storage);

  bp::scope in_history =
      bp::class_<history_callback_t, bp::bases<CallbackBase>>(
          "HistoryCallback", "Store the history of solver's variables.",
          bp::init<bool, bool, bool>((bp::arg("self"),
                                      bp::arg("store_pd_vars") = true,
                                      bp::arg("store_values") = true,
                                      bp::arg("store_residuals") = true)))
          .def_readonly("storage", &helpers::history_callback<Scalar>::storage);

  bp::class_<history_storage_t>("history_storage")
      .def_readonly("xs", &history_storage_t::xs)
      .def_readonly("us", &history_storage_t::us)
      .def_readonly("lams", &history_storage_t::lams)
      .def_readonly("values", &history_storage_t::values)
      .def_readonly("merit_values", &history_storage_t::merit_values)
      .def_readonly("prim_infeas", &history_storage_t::prim_infeas)
      .def_readonly("dual_infeas", &history_storage_t::dual_infeas)
      .def_readonly("al_iters", &history_storage_t::al_index)
      .def_readonly("prim_tols", &history_storage_t::prim_tols)
      .def_readonly("dual_tols", &history_storage_t::dual_tols);

  StdVectorPythonVisitor<std::vector<context::VectorOfVectors>, true>::expose(
      "StdVecVec_VectorXs", "std::vector of std::vector of Eigen::MatrixX.");
}

void exposeCallbacks() {
  bp::register_ptr_to_python<shared_ptr<CallbackBase>>();

  bp::class_<CallbackWrapper, boost::noncopyable>("BaseCallback",
                                                  "Base callback for solvers.",
                                                  bp::init<>(bp::args("self")))
      .def("call", bp::pure_virtual(&CallbackWrapper::call),
           bp::args("self", "workspace", "results"));

  exposeHistoryCallback();
  using LSCallback = LinesearchCallback<Scalar>;
  using LSData = LSCallback::Data;
  eigenpy::enableEigenPySpecific<LSData::Matrix2Xs>();
  bp::class_<LSCallback, bp::bases<CallbackBase>>("LinesearchCallback",
                                                  bp::init<>(bp::args("self")))
      .def_readwrite("alpha_min", &LSCallback::alpha_min)
      .def_readwrite("alpha_max", &LSCallback::alpha_max)
      .def(
          "get",
          +[](const LSCallback &m, std::size_t t) {
            if (t >= m.storage_.size()) {
              PyErr_SetString(
                  PyExc_IndexError,
                  fmt::format("Index {} is out of bounds.", t).c_str());
              bp::throw_error_already_set();
            }
            return m.get(t);
          },
          bp::args("self", "t"))
      .def("get_d", &LSCallback::get_dphi, bp::args("self", "t"));
}
} // namespace python
} // namespace proxddp
