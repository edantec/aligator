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
from diffsim_rs_utils import DiffSimDynamicsModel, RSCallback, DiffSimDynamicsModelAugmented
from diffsim.simulator import Simulator, SimulatorNode

from robot_properties_teststand.config import TeststandConfig

teststand_config = TeststandConfig()
rmodel, rgeom_model, rvisual_model = TeststandConfig.create_solo_leg_model()


nq = rmodel.nq
nv = rmodel.nv
ROT_NULL = np.eye(3)
run_name = "solo_foot_up"

q_init = 1*teststand_config.q0
# q_init = pin.neutral(rmodel)
# q_init[0] = .33
# q_init[1] = 0.2
# q_init[2] = -0.4

if 0:
    vizer = init_viewer_ellipsoids(rmodel, rgeom_model, rgeom_model, open=True)
    # streched configuration
    # q_init = pin.neutral(rmodel)
    # q_init[0] = .335

    # small angle
    # q_init = pin.neutral(rmodel)
    # q_init[0] = .33
    # q_init[1] = 0.2
    # q_init[2] = -0.4


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

    def process_args(self):
        if self.record:
            self.display = True

def main(args: Args):

    os.makedirs("assets", exist_ok=True)
    print(args)

    if args.display:
        # 1st arg is the plane normal
        # 2nd arg is offset from origin
        plane = fcl.Plane(np.array([0.0, 0.0, 1.0]), 0.0)
        plane_obj = pin.GeometryObject("plane", 0, pin.SE3.Identity(), plane)
        plane_obj.meshColor[:] = [1.0, 1.0, 0.95, 1.0]
        plane_obj.meshScale[:] = 2.0
        rvisual_model.addGeometryObject(plane_obj)
        rgeom_model.addGeometryObject(plane_obj)

    def add_objective_vis_models(x_tar):
        """Add visual guides for the objectives."""
        objective_color = np.array([5, 104, 143, 200]) / 255.0
        sp1_obj = pin.GeometryObject(
            "obj1", 0, pin.SE3(ROT_NULL, x_tar[:3]), fcl.Sphere(5.)
        )
        sp1_obj.meshColor[:] = objective_color
        rvisual_model.addGeometryObject(sp1_obj)

    nx = 2*rmodel.nv
    nu = nv - 1
    if args.augmented:
        space = manifolds.VectorSpace(nx + 2*nu)
    else:
        space = manifolds.VectorSpace(nx)

    act_matrix = np.eye(nv, nu, -1)

    N_samples_init = 1
    noise_intensity_init = 0.
    max_rsddp_iter = 1
    max_iters = 100

    coeff_friction = 0.7
    coeff_rest = 0.0
    # dt = 0.01
    # Tf = 1.5
    dt = 5e-3
    Tf = 0.1
    nsteps = int(Tf / dt)
    print("nsteps: {:d}".format(nsteps))

    if args.augmented:
        dynmodel = DiffSimDynamicsModelAugmented(space, rmodel, act_matrix, rgeom_model, coeff_friction, coeff_rest, nu, dt)
    else:
        dynmodel = DiffSimDynamicsModel(space, rmodel, act_matrix, rgeom_model, coeff_friction, coeff_rest,nu, dt)


    q0 = 1*q_init
    dq0 = pin.difference(rmodel, rmodel.qref, q0)
    x0 = np.concatenate([dq0, np.zeros(nv)])

    tau = pin.rnea(rmodel, rmodel.createData(), q0, np.zeros(nv), np.zeros(nv))
    u0, _, _, _ = np.linalg.lstsq(act_matrix, tau)
    u0 = np.zeros(nu)

    if False:
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
        

    def constraint_quasistatic_torque(nodes, x0, u0, S):
        B = torch.zeros((nv, nv))  # B is the actuation matrix when u is control including FF
        B[1:, 1:] = torch.eye(nv - 1)
        def compute_static_torque(node, x, u, S):
            # print(f"x: {x}, u: {u}")
            x_next = node.makeStep(torch.tensor(x), torch.tensor(S @ u), calcPinDiff=True)
            model = node.rmodel
            data = model.createData()
            q = node.rdata.q
            # print(f"q: {q}")

            u_static = pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv))  # len(u_static) = 14
            Jn = node.rdata.Jn_
            Jt = node.rdata.Jt_
            A = (torch.vstack((B, Jn, Jt)).T).detach().numpy()
            tau_lamN_lamT = np.linalg.pinv(A) @ u_static

            tau_static = tau_lamN_lamT[:model.nv]
            tau_static_ = torch.tensor(tau_static)
            # lamN_static = tau_lamN_lamT[model.nv:model.nv+4]
            # lamT_static = tau_lamN_lamT[model.nv+4:]

            # print(f"tau_static: {tau_static}")
            # print(f"lamN_static: {lamN_static}")
            # print(f"lamT_static: {lamT_static}")
            return tau_static_[1:].detach().numpy(), x_next.detach().numpy()

        u_list, x_list = [], [x0]
        first_node = nodes[0]
        u, _ = compute_static_torque(first_node, x0, u0, S)

        x = 1*x0
        for i, node in enumerate(nodes):
            u, x = compute_static_torque(node, x, u, S)
            u_list.append(1*u)
            x_list.append(1*x)
        return u_list, x_list

    sim_nodes = [SimulatorNode(Simulator(rmodel, rgeom_model, dt, coeff_friction, coeff_rest, dt_collision=dt)) for _ in range(nsteps)]

    # compute initial guess
    us_init, xs_init = constraint_quasistatic_torque(
        sim_nodes, x0, u0, act_matrix
    )

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

    # us_init = [u0] * nsteps
    # xs_init = [x0] * (nsteps + 1)

    x_tar = 1*x0
    x_tar[0] += 0.3  # go up

    if args.augmented:
        x_tar = np.concatenate([x_tar, np.zeros(nu), np.zeros(nu)])

    add_objective_vis_models(np.array([0.0, 0.0, x_tar[0]]))

    u_max = rmodel.effortLimit[1:]*1e-2
    u_min = -1*u_max

    times = np.linspace(0, Tf, nsteps + 1)

    print(f"RUN: {run_name} + target: {x_tar} + init: {q0}")

    def get_task():
        weights = np.zeros(space.ndx)
        weights[0:1] = 1.0
        # weights[1:nv] = 1e-3
        weights[nv+1:2*nv] = 5e-3
        weights[nv+1:2*nv] = 1e-3
        if args.augmented:
            weights[2*nv+nu:] = 1e-3
        # print(f"weight: {weights}")
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

            weights, x_tar = get_task()
            weights = np.zeros(space.ndx)
            weights[nv+1:2*nv] = 5e-3

            if args.augmented:
                weights[2*nv+nu:] = 1e-4

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

        weights, x_tar = get_task()

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


    TAG = "quadrotor_takeoff"

    root_pt_opt = np.stack(xs_opt)[:, :1]
    if args.plot:
        if len(results.lams) > 0:
            plot_costate_value()

        nplot = 3
        fig: plt.Figure = plt.figure(figsize=(9.6, 5.4))
        ax0: plt.Axes = fig.add_subplot(1, nplot, 1)
        ax0.plot(times[:-1], us_opt)
        ax0.hlines((u_min[0], u_max[0]), *times[[0, -1]], colors="k", alpha=0.3, lw=1.4)
        ax0.set_title("Controls")
        ax0.set_xlabel("Time")
        ax1: plt.Axes = fig.add_subplot(1, nplot, 2)
        # ax1.plot(times, root_pt_opt)
        ax1.plot(times, root_pt_opt)
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
            vizer.display(qs_opt[0])
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
