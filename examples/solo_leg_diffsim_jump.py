"""
Simple quadrotor dynamics example.

Inspired by: https://github.com/loco-3d/crocoddyl/blob/master/examples/quadrotor.py
"""
import os
import pinocchio as pin
import hppfcl as fcl
import example_robot_data as erd
import torch
import numpy as np
import matplotlib.pyplot as plt
import proxddp
import meshcat
import time

from proxnlp import constraints
from proxddp import manifolds
from utils import ArgsBase

from diffsim.utils_render import init_viewer_ellipsoids
from diffsim_rs_utils import DiffSimDynamicsModel, RSCallback, DiffSimDynamicsModelAugmented, constraint_quasistatic_torque_diffsim, constraint_quasistatic_torque_contact_bench
from diffsim.simulator import Simulator, SimulatorNode

from robot_properties_teststand.config import TeststandConfig

np.random.seed(1234)
torch.manual_seed(1234)
torch.set_default_dtype(torch.float64)

teststand_config = TeststandConfig()
rmodel, (rgeom_model, rgeom_model_cb), rvisual_model = TeststandConfig.create_solo_leg_model()

nq = rmodel.nq
nv = rmodel.nv
ROT_NULL = np.eye(3)
run_name = "solo_foot_up"

# Choose init configuration
q_init = 1*teststand_config.q0

# streched configuration
# q_init = pin.neutral(rmodel)
# q_init[0] = .335

# small angle
# q_init = pin.neutral(rmodel)
# q_init[0] = .33
# q_init[1] = 0.2
# q_init[2] = -0.4

if 0:
    vizer = init_viewer_ellipsoids(rmodel, rgeom_model, rgeom_model, open=True)
    vizer.display(q_init)
    input("Displaying q_init, press enter to visualize target configuration")
    q_target = 1*q_init
    q_target[0] += .02
    vizer.display(q_target)
    input()
    exit

class Args(ArgsBase):
    integrator = "semieuler"
    bounds: bool = False
    """Use control bounds"""
    plot: bool = False  # Plot the trajectories
    display: bool = False
    term_cstr: bool = False
    proxddp: bool = False
    augmented: bool = False
    rsddp: bool = False
    initial_guess: str = "diffsim_qp"

    def process_args(self):
        if self.record:
            self.display = True

def main(args: Args):

    os.makedirs("assets", exist_ok=True)
    print(args)

    def add_objective_vis_models(x_tar):
        """Add visual guides for the objectives."""
        objective_color = np.array([5, 104, 143, 200]) / 255.0
        sp1_obj = pin.GeometryObject(
            "obj1", 0, pin.SE3(ROT_NULL, x_tar[:3]), fcl.Sphere(0.05)
        )
        sp1_obj.meshColor[:] = objective_color
        rgeom_model.addGeometryObject(sp1_obj)

    nx = 2*rmodel.nv
    nu = nv - 1

    if args.augmented:
        space = manifolds.VectorSpace(nx + 2*nu)
    else:
        space = manifolds.VectorSpace(nx)

    act_matrix = np.eye(nv, nu, -1)

    N_samples_init = 1 if not args.rsddp else 4
    noise_intensity_init = 0. if not args.rsddp else 0.00005
    max_rsddp_iter = 1
    max_iters = 200

    coeff_friction = 0.7
    coeff_rest = 0.0
    dt = 5e-3
    Tf = 0.2
    nsteps = int(Tf / dt)
    print("nsteps: {:d}".format(nsteps))

    if args.augmented:
        dynmodel = DiffSimDynamicsModelAugmented(space, rmodel, act_matrix, rgeom_model, coeff_friction, coeff_rest, nu, dt)
    else:
        dynmodel = DiffSimDynamicsModel(space, rmodel, act_matrix, rgeom_model, coeff_friction, coeff_rest,nu, dt)

    q0 = 1*q_init
    dq0 = pin.difference(rmodel, rmodel.qref, q0)
    x0 = np.concatenate([dq0, np.zeros(nv)])

    x_tar = 1*x0
    x_tar[0] += 0.3  # go up

    u0 = np.zeros(nu)

    if False:
        # Run a simple forward simulation to check behavior of the simulator
        sim_nodes = [SimulatorNode(Simulator(rmodel, rgeom_model, dt, coeff_friction, coeff_rest, dt_collision=dt)) for _ in range(nsteps)]
        x = torch.tensor(x0)
        u = torch.zeros(3)
        q_list = []
        for i, node in enumerate(sim_nodes):
            x = node.makeStep(x, u, calcPinDiff=True)
            q_list.append(1*node.rdata.q)

        vizer = init_viewer_ellipsoids(rmodel, rgeom_model,rvisual_model, open=True)
        for q in q_list:
            vizer.display(q)
            time.sleep(0.01)
        exit()
        

    # compute initial guess for control to keep system in static equilibrium
    sim_nodes = [SimulatorNode(Simulator(rmodel, rgeom_model, dt, coeff_friction, coeff_rest, dt_collision=dt)) for _ in range(nsteps)]
    if args.initial_guess == "diffsim_qp":
        us_init, xs_init = constraint_quasistatic_torque_diffsim(
            sim_nodes, x0, act_matrix, version="qp"
        )
    elif args.initial_guess == "diffsim_lstsq":
        us_init, xs_init = constraint_quasistatic_torque_diffsim(
            sim_nodes, x0, act_matrix, version="lstsq"
        )
    elif args.initial_guess == "cb_qp":
        us_init, xs_init = constraint_quasistatic_torque_contact_bench(
            rmodel, rgeom_model_cb, x0, act_matrix, len(sim_nodes), dt, dynmodel.sim.sim.k_baumgarte, version="qp"
        )
    elif args.initial_guess == "cb_lstsq":
        us_init, xs_init = constraint_quasistatic_torque_contact_bench(
            rmodel, rgeom_model_cb, x0, act_matrix, len(sim_nodes), dt, dynmodel.sim.sim.k_baumgarte, version="lstsq"
        )
    else:
        raise ValueError(f"Unknown initial guess: {args.initial_guess}")



    if args.augmented:
        xs_init_augmented = []
        for i, x_init in enumerate(xs_init):
            if i == 0 and i == 1:
                x_init = np.concatenate([x_init, us_init[i], np.zeros(nu)])
            else:
                x_init = np.concatenate([x_init, us_init[i-1], us_init[i-1]-us_init[i-2]])
            xs_init_augmented.append(x_init)
        xs_init = xs_init_augmented
       
    if False:
        # Run a simple forward simulation to check if the initial guess is correct to keep the system in static equilibrium
        sim_nodes = [SimulatorNode(Simulator(rmodel, rgeom_model, dt, coeff_friction, coeff_rest, dt_collision=dt)) for _ in range(nsteps)]
        x = torch.tensor(x0)
        q_list = []
        for i, node in enumerate(sim_nodes):
            u = torch.tensor(act_matrix @ us_init[i])
            x = node.makeStep(x, u, calcPinDiff=True)
            q_list.append(1*node.rdata.q)

        vizer = init_viewer_ellipsoids(rmodel, rgeom_model,rvisual_model, open=True)
        for q in q_list:
            vizer.display(q)
            print(f"q: {q}")
            time.sleep(0.1)
        exit()

    if args.augmented:
        x_tar = np.concatenate([x_tar, np.zeros(nu), np.zeros(nu)])

    add_objective_vis_models(np.array([0.0, 0.0, q_init[0] + x_tar[0]]))

    u_max = rmodel.effortLimit[1:]
    u_min = -1*u_max

    print(f"RUN: {run_name} + target: {x_tar} + init: {q0}")

    def get_task(task: str):
        weights = np.zeros(space.ndx)
        if task == "running":
            weights[nv+1:2*nv] = 1e-4  # only penalize velocity of joints (not z position/velocity)

        elif task == "terminal":
            weights[0:1] = 1.0  # only penalize z position, not velocity
            weights[nv+1:2*nv] = 5e-3
        
        if args.augmented:
            # penalize the augmented state, namely the the difference of the control (not the control itself)
            # weights[2*nv+nu:] = 1e-3
            weights[2*nv+nu:] = 1e-4
        print(f"{task} weights: {weights}")
        return weights, x_tar

    def setup():
        w_u = np.eye(nu) * 1e-3
        stages = []
        if args.bounds:
            u_identity_fn = proxddp.ControlErrorResidual(space.ndx, np.zeros(nu))
            box_set = constraints.BoxConstraint(u_min, u_max)
            ctrl_cstr = proxddp.StageConstraint(u_identity_fn, box_set)

        for i in range(nsteps):
            rcost = proxddp.CostStack(space, nu)

            weights, x_tar = get_task("running")
            xreg_cost = proxddp.QuadraticStateCost(
                space, nu, x_tar, np.diag(weights) * dt
            )
            rcost.addCost(xreg_cost)
            ureg_cost = proxddp.QuadraticControlCost(space, nu, w_u * dt)
            ureg_cost.target = us_init[0]
            rcost.addCost(ureg_cost)

            stage = proxddp.StageModel(rcost, dynmodel)
            if args.bounds:
                stage.addConstraint(ctrl_cstr)
            stages.append(stage)

        weights, x_tar = get_task("terminal")

        if not args.term_cstr:
            weights *= 10.0
        term_cost = proxddp.QuadraticStateCost(space, nu, x_tar, np.diag(weights))
        prob = proxddp.TrajOptProblem(xs_init[0], stages, term_cost=term_cost)
        if args.term_cstr:
            term_cstr = proxddp.StageConstraint(
                proxddp.StateErrorResidual(space, nu, x_tar),
                constraints.EqualityConstraintSet(),
            )
            prob.addTerminalConstraint(term_cstr)
        return prob

    problem = setup()

    tol = 1e-3
    verbose = proxddp.VerboseLevel.VERBOSE
    history_cb = proxddp.HistoryCallback()
    rs_cb = RSCallback(N_samples_init, noise_intensity_init)
    solver = proxddp.SolverFDDP(tol, verbose=verbose)
    if args.proxddp:
        mu_init = 1e-1
        rho_init = 0.0
        solver = proxddp.SolverProxDDP(tol, mu_init, rho_init, verbose=verbose)
    solver.max_iters = max_iters
    solver.registerCallback("his", history_cb)
    solver.registerCallback("rs", rs_cb)
    solver.setup(problem)

    results = solver.getResults()
    workspace = solver.getWorkspace()
    solver.getCallback("rs").call(workspace, results)  # init noise

    for i in range(max_rsddp_iter):
        print("current noise:", solver.getCallback("rs").noise_intensity)
        solver.run(problem, xs_init, us_init)
        if solver.getCallback("rs").noise_intensity <1e-3:
            break
        else:
            results = solver.getResults()
            xs_init = results.xs
            us_init = results.us
            solver.getCallback("rs").noise_intensity /= 2.

    results = solver.getResults()
    workspace = solver.getWorkspace()

    print(results)

    xs_opt = results.xs.tolist()
    us_opt = results.us.tolist()
    
    val_grad = [vp.Vx for vp in workspace.value_params]

    def plot_costate_value() -> plt.Figure:
        lams_stack = np.stack([la[: space.ndx] for la in results.lams]).T
        costate_stack = lams_stack[:, 1 : nsteps + 1]
        vx_stack = np.stack(val_grad).T[:, 1:]
        plt.figure()
        plt.subplot(131)
        mmin = min(np.min(costate_stack), np.min(vx_stack))
        mmax = max(np.max(costate_stack), np.max(vx_stack))
        plt.imshow(costate_stack, vmin=mmin, vmax=mmax, aspect="auto")
        # plt.vlines(idx_switch, *plt.ylim(), colors="r", label="switch")
        plt.legend()

        plt.xlabel("Time $t$")
        plt.ylabel("Dimension")
        plt.title("Multipliers")
        plt.subplot(132)
        plt.imshow(vx_stack, vmin=mmin, vmax=mmax, aspect="auto")
        plt.colorbar()
        plt.xlabel("Time $t$")
        plt.ylabel("Dimension")
        plt.title("$\\nabla_xV$")

        plt.subplot(133)
        err = np.abs(costate_stack - vx_stack)
        plt.imshow(err, cmap="Reds", aspect="auto")
        plt.title("$\\lambda - V'_x$")
        plt.colorbar()
        plt.tight_layout()
        return plt.gcf()


    TAG = "solo_leg_jump"

    dz_position = np.stack(xs_opt)[:, 0:1]

    if args.plot:
        times = np.linspace(0, Tf, nsteps + 1)

        if len(results.lams) > 0:
            plot_costate_value()

        nplot = 3
        fig: plt.Figure = plt.figure(figsize=(9.6, 5.4))
        ax0: plt.Axes = fig.add_subplot(1, nplot, 1)
        ax0.plot(times[:-1], us_opt)
        # ax0.hlines((u_min[0], u_max[0]), *times[[0, -1]], colors="k", alpha=0.3, lw=1.4)
        ax0.set_title(f"Controls in [{u_min[0]:.2f}, {u_max[0]:.2f}]")
        ax0.set_xlabel("Time")
        ax1: plt.Axes = fig.add_subplot(1, nplot, 2)
        ax1.plot(times, dz_position)
        ax1.hlines(x_tar[0], *times[[0, -1]], colors="k", alpha=0.3, lw=1.4)
        plt.legend(["$z$"])
        # ax1.scatter([times_wp[-1]] * 3, x_term[:3], marker=".", c=["C0", "C1", "C2"])
        ax2: plt.Axes = fig.add_subplot(1, nplot, 3)
        n_iter = np.arange(len(history_cb.storage.prim_infeas.tolist()))
        ax2.semilogy(
            n_iter[1:], history_cb.storage.prim_infeas.tolist()[1:], label="Primal err."
        )
        ax2.semilogy(n_iter, history_cb.storage.dual_infeas.tolist(), label="Dual err.")
        ax2.set_xlabel("Iterations")
        ax2.legend()

        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig("assets/{}.{}".format(TAG, ext))
        plt.show()

    if args.display:
        vizer = init_viewer_ellipsoids(rmodel, rgeom_model, rvisual_model, open=True)
        vizer.display(pin.integrate(rmodel,rmodel.qref,x0[:nv]))

        cam_dist = 2.0
        directions_ = [np.array([1.0, 1.0, 0.5])]
        directions_.append(np.array([1.0, -1.0, 0.8]))
        directions_.append(np.array([0.1, 0.1, 1.0]))
        directions_.append(np.array([0.0, -1.0, 0.8]))
        for d in directions_:
            d /= np.linalg.norm(d)

        vid_uri = "assets/{}.mp4".format(TAG)
        qs_opt = [pin.integrate(rmodel, rmodel.qref,x[:nv]) for x in xs_opt]
        base_link_id = rmodel.getFrameId("base_link")

        def get_callback(i: int):
            def _callback(t):
                n = len(root_pt_opt)
                n = min(t, n)
                rp = root_pt_opt[n]
                pos = rp + directions_[i] * cam_dist
                vizer.setCameraPosition(pos)
                vizer.setCameraTarget(rp)
                vel = xs_opt[t][nv:]
                pin.forwardKinematics(rmodel, vizer.data, qs_opt[t], vel)
                vizer.drawFrameVelocities(base_link_id)

            return _callback

        input("[enter to play]")
        while True:
            vizer.play(qs_opt, dt*3)
            vizer.display(qs_opt[-1])
            a = input("press to continue, [q] to quit")
            if a == "q":
                break

        if args.record:
            ctx = vizer.create_video_ctx(vid_uri, fps=30)
        else:
            import contextlib

            ctx = contextlib.nullcontext()
        with ctx:
            for i in range(4):
                vizer.play(qs_opt, dt, get_callback(i))


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
