#include "proxddp/python/fwd.hpp"

#include "proxddp/core/solver-proxddp.hpp"

namespace proxddp {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(prox_run_overloads, run, 1, 4)

void exposeProxDDP() {
  using context::Scalar;
  using context::TrajOptProblem;
  using Workspace = WorkspaceTpl<Scalar>;
  using Results = ResultsTpl<Scalar>;

  bp::class_<Workspace, bp::bases<WorkspaceBaseTpl<Scalar>>>(
      "Workspace", "Workspace for ProxDDP.",
      bp::init<const TrajOptProblem &>(bp::args("self", "problem")))
      .def_readonly("prox_datas", &Workspace::prox_datas)
      .def_readonly("kkt_mat", &Workspace::kkt_mats_)
      .def_readonly("kkt_rhs", &Workspace::kkt_rhs_)
      .def_readonly("trial_lams", &Workspace::trial_lams)
      .def_readonly("inner_crit", &Workspace::inner_criterion)
      .def_readonly("stage_prim_infeas", &Workspace::stage_prim_infeas)
      .def_readonly("stage_dual_infeas", &Workspace::stage_dual_infeas)
      .def_readonly("stage_inner_crits", &Workspace::stage_inner_crits)
      .def("cycle_append", &Workspace::cycle_append,
           bp::args("self", "stage_model"),
           "From a StageModel object, allocate its data object, rotate the "
           "workspace (using `cycle_left()`) and insert the allocated data "
           "(useful for MPC).")
      .def(PrintableVisitor<Workspace>());

  bp::class_<Results, bp::bases<ResultsBaseTpl<Scalar>>>(
      "Results", "Results struct for proxDDP.",
      bp::init<const TrajOptProblem &>())
      .def_readonly("al_iter", &Results::al_iter);

  using SolverType = SolverProxDDP<Scalar>;

  bp::enum_<MultiplierUpdateMode>(
      "MultiplierUpdateMode", "Enum for the kind of multiplier update to use.")
      .value("NEWTON", MultiplierUpdateMode::NEWTON)
      .value("PRIMAL", MultiplierUpdateMode::PRIMAL)
      .value("PRIMAL_DUAL", MultiplierUpdateMode::PRIMAL_DUAL);

  bp::enum_<LinesearchMode>("LinesearchMode", "Linesearch mode.")
      .value("PRIMAL", LinesearchMode::PRIMAL)
      .value("PRIMAL_DUAL", LinesearchMode::PRIMAL_DUAL);

  bp::enum_<RolloutType>("RolloutType", "Rollout type.")
      .value("ROLLOUT_LINEAR", RolloutType::LINEAR)
      .value("ROLLOUT_NONLINEAR", RolloutType::NONLINEAR)
      .export_values();

  bp::enum_<HessianApprox>("HessianApprox",
                           "Level of approximation for te Hessian.")
      .value("HESSIAN_EXACT", HessianApprox::EXACT)
      .value("HESSIAN_GAUSS_NEWTON", HessianApprox::GAUSS_NEWTON)
      .export_values();

  bp::class_<SolverType, boost::noncopyable>(
      "SolverProxDDP",
      "A proximal, augmented Lagrangian solver, using a DDP-type scheme to "
      "compute "
      "search directions and feedforward, feedback gains."
      " The solver instance initializes both a Workspace and Results which "
      "can "
      "be retrieved"
      " through the `getWorkspace` and `getResults` methods, respectively.",
      bp::init<Scalar, Scalar, Scalar, std::size_t, VerboseLevel,
               HessianApprox>(
          (bp::arg("self"), bp::arg("tol"), bp::arg("mu_init") = 1e-2,
           bp::arg("rho_init") = 0., bp::arg("max_iters") = 1000,
           bp::arg("verbose") = VerboseLevel::QUIET,
           bp::arg("hess_approx") = HessianApprox::GAUSS_NEWTON)))
      .def_readwrite("bcl_params", &SolverType::bcl_params, "BCL parameters.")
      .def_readwrite("is_x0_fixed", &SolverType::is_x0_fixed,
                     "Set x0 to be fixed to the initial condition.")
      .def_readwrite("multiplier_update_mode",
                     &SolverType::multiplier_update_mode)
      .def_readwrite("mu_init", &SolverType::mu_init,
                     "Initial AL penalty parameter.")
      .def_readwrite("rho_init", &SolverType::rho_init,
                     "Initial proximal regularization.")
      .def_readwrite(
          "mu_dyn_scale", &SolverType::mu_dyn_scale,
          "Scale factor for the dynamics' augmented Lagrangian penalty.")
      .def_readwrite(
          "mu_stage_scale", &SolverType::mu_stage_scale,
          "Scale factor for the AL penalty on stagewise constraints.")
      .def_readwrite("mu_min", &SolverType::MU_MIN,
                     "Lower bound on the AL penalty parameter.")
      .def_readwrite("ls_mode", &SolverType::ls_mode, "Linesearch mode.")
      .def_readwrite("rollout_type", &SolverType::rollout_type, "Rollout type.")
      .def_readwrite("dual_weight", &SolverType::dual_weight,
                     "Dual penalty weight.")
#ifndef NDEBUG
      .def_readwrite("dump_linesearch_plot", &SolverType::dump_linesearch_plot,
                     "[Debug] Dump a plot of the linesearch support function "
                     "at every iteration.")
#endif
      .def(SolverVisitor<SolverType>())
      .def("run", &SolverType::run,
           prox_run_overloads(
               (bp::arg("self"), bp::arg("problem"), bp::arg("xs_init"),
                bp::arg("us_init"), bp::arg("lams_init")),
               "Run the algorithm. Can receive initial guess for "
               "multiplier trajectory."));
}

} // namespace python
} // namespace proxddp
