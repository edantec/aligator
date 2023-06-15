"""
Simple quadrotor dynamics example.

Inspired by: https://github.com/loco-3d/crocoddyl/blob/master/examples/quadrotor.py
"""
import pinocchio as pin
import hppfcl as fcl
import numpy as np
import matplotlib.pyplot as plt
import os
import proxddp
import meshcat

from proxddp import manifolds
from proxnlp import constraints

from diffsim_rs_utils import create_quadrotor_model, DiffSimDyamicsModel, RSCallback

from utils import ArgsBase
import torch

torch.manual_seed(1234)
np.random.seed(1234)

rmodel, rgeom_model, rvisual_model, rdata, rgeom_data, _ = create_quadrotor_model()
nq = rmodel.nq
nv = rmodel.nv
ROT_NULL = np.eye(3)

class Args(ArgsBase):
    integrator = "semieuler"
    bounds: bool = False
    """Use control bounds"""
    plot: bool = False  # Plot the trajectories
    display: bool = False
    term_cstr: bool = False
    proxddp: bool = False

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

    def add_objective_vis_models(x_tar1):
        """Add visual guides for the objectives."""
        objective_color = np.array([5, 104, 143, 200]) / 255.0
        sp1_obj = pin.GeometryObject(
            "obj1", 0, pin.SE3(ROT_NULL, x_tar1[:3]), fcl.Sphere(0.05)
        )
        sp1_obj.meshColor[:] = objective_color
        rvisual_model.addGeometryObject(sp1_obj)

    rgeom_model.geometryObjects[0].geometry.computeLocalAABB()
    quad_radius = rgeom_model.geometryObjects[0].geometry.aabb_radius

    # space = manifolds.MultibodyPhaseSpace(rmodel)
    nx = 2*rmodel.nv
    space = manifolds.VectorSpace(nx)

    # The matrix below maps rotor controls to torques

    d_cog, cf, cm, u_lim, _ = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
    QUAD_ACT_MATRIX = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, d_cog, 0.0, -d_cog],
            [-d_cog, 0.0, d_cog, 0.0],
            [-cm / cf, cm / cf, -cm / cf, cm / cf],
        ]
    )
    nu = QUAD_ACT_MATRIX.shape[1]  # = no. of nrotors

    # ode_dynamics = proxddp.dynamics.MultibodyFreeFwdDynamics(space, QUAD_ACT_MATRIX)

    coeff_friction = 0.9
    coeff_rest = 0.0
    dt = 0.01
    Tf = 1.5
    nsteps = int(Tf / dt)
    print("nsteps: {:d}".format(nsteps))

    dynmodel = DiffSimDyamicsModel(space, rmodel, QUAD_ACT_MATRIX, rgeom_model, coeff_friction, coeff_rest,nu, dt)

    q0 = 1*rmodel.qref
    q0[2] = 1.87707555e-01 
    dq0 = pin.difference(rmodel, rmodel.qref, q0)
    x0 = np.concatenate([dq0, np.zeros(nv)])

    tau = pin.rnea(rmodel, rdata, q0, np.zeros(nv), np.zeros(nv))
    u0, _, _, _ = np.linalg.lstsq(QUAD_ACT_MATRIX, tau)
    # u0 = np.zeros(nu)

    us_init = [u0] * nsteps
    xs_init = [x0] * (nsteps + 1)

    x_tar1 = space.neutral()
    x_tar1[:3] = (0., 0., 1.0)
    add_objective_vis_models(x_tar1)

    u_max = u_lim * np.ones(nu)
    u_min = np.zeros(nu)

    times = np.linspace(0, Tf, nsteps + 1)

    def get_task_schedule():
        weights1 = np.zeros(space.ndx)
        weights1[:3] = 4.0
        weights1[3:6] = 1e-1
        weights1[nv:] = 1e-3

        def weight_target_selector(i):
            x_tar = x_tar1
            weights = weights1
            return weights, x_tar

        return weight_target_selector

    task_schedule = get_task_schedule()

    def setup():
        w_u = np.eye(nu) * 1e-3
        stages = []
        if args.bounds:
            u_identity_fn = proxddp.ControlErrorResidual(space.ndx, np.zeros(nu))
            box_set = constraints.BoxConstraint(u_min, u_max)
            ctrl_cstr = proxddp.StageConstraint(u_identity_fn, box_set)

        for i in range(nsteps):
            rcost = proxddp.CostStack(space, nu)

            weights, x_tar = task_schedule(i)

            xreg_cost = proxddp.QuadraticStateCost(
                space, nu, x_tar, np.diag(weights) * dt
            )
            rcost.addCost(xreg_cost)
            ureg_cost = proxddp.QuadraticControlCost(space, nu, w_u * dt)
            rcost.addCost(ureg_cost)

            stage = proxddp.StageModel(rcost, dynmodel)
            if args.bounds:
                stage.addConstraint(ctrl_cstr)
            stages.append(stage)

        weights, x_tar = task_schedule(nsteps)
        if not args.term_cstr:
            weights *= 10.0
        term_cost = proxddp.QuadraticStateCost(space, nu, x_tar, np.diag(weights))
        prob = proxddp.TrajOptProblem(x0, stages, term_cost=term_cost)
        if args.term_cstr:
            term_cstr = proxddp.StageConstraint(
                proxddp.StateErrorResidual(space, nu, x_tar),
                constraints.EqualityConstraintSet(),
            )
            prob.addTerminalConstraint(term_cstr)
        return prob

    _, x_term = task_schedule(nsteps)
    problem = setup()

    viewer = meshcat.Visualizer()
    vizer = pin.visualize.MeshcatVisualizer(
        rmodel, rgeom_model, rvisual_model, data=rdata
    )
    vizer.initViewer(viewer, loadModel=True, open=args.display)
    vizer.displayCollisions(True)
    # vizer.display(q0)
    vizer.display(pin.integrate(rmodel,rmodel.qref,x0[:nv]))
    # input()
    # vizer.display(x_tar1[:nq])
    # input()


    N_samples_init = 4
    noise_intensity_init = 1.
    max_rsddp_iter = 3
    tol = 1e-3
    verbose = proxddp.VerboseLevel.VERBOSE
    history_cb = proxddp.HistoryCallback()
    rs_cb = RSCallback(N_samples_init, noise_intensity_init)
    solver = proxddp.SolverFDDP(tol, verbose=verbose)
    if args.proxddp:
        mu_init = 1e-1
        rho_init = 0.0
        solver = proxddp.SolverProxDDP(tol, mu_init, rho_init, verbose=verbose)
    solver.max_iters = 6
    solver.registerCallback("his", history_cb)
    solver.registerCallback("rs", rs_cb)
    solver.setup(problem)
    results = solver.getResults()
    workspace = solver.getWorkspace()
    solver.getCallback("rs").call(workspace, results)

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

    root_pt_opt = np.stack(xs_opt)[:, :3]
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
        ax1.plot(times, root_pt_opt)
        plt.legend(["$x$", "$y$", "$z$"])
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