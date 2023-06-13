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

from proxnlp import constraints
from proxddp import manifolds
from proxddp.dynamics import ExplicitDynamicsModel, ExplicitDynamicsData
from utils import ArgsBase

from diffsim.simulator import Simulator, SimulatorNode
from diffsim.shapes import Plane, Ellipsoid
from diffsim.collision_pairs import CollisionPairPlaneEllipsoid, CollisionPairEllipsoidEllipsoid
from diffsim.utils_render import init_viewer_ellipsoids

from pinocchio.visualize import MeshcatVisualizer

def create_solo_model():
   # robot model
    robot = erd.load('solo12')
    rmodel = robot.model.copy()
    rdata = rmodel.createData()
    rmodel.qref = rmodel.referenceConfigurations["standing"]
    rmodel.qinit = rmodel.referenceConfigurations["standing"]
    # Geometry model
    rgeomModel = robot.collision_model
    # add feet
    a = 0.01910275 #1730829468
    rFoot = np.array([0.2, 0.08, 0.05])
    R = np.eye(3)
    tLeft = np.array([0, 0.008,  -0.16])
    MLeft = pin.SE3(R, tLeft)
    tRight = np.array([0, -0.008,  -0.16])
    MRight = pin.SE3(R, tRight)
    r = np.array([a, a, a])  
    
    rgeomModel.computeColPairDist = []

    n = np.array([0., 0., 1])
    p = np.array([0., 0., 0.0])
    h = np.array([100., 100., 0.01])
    plane_shape = Plane(0, 'plane', n, p, h)
    T = pin.SE3(plane_shape.R, plane_shape.t)
    plane = pin.GeometryObject("plane", 0, 0, plane_shape, T)
    plane.meshColor = np.array([0.5, 0.5, 0.5, 1.]) 
    planeId = rgeomModel.addGeometryObject(plane)
    
    frames_names = ["HR_FOOT","HL_FOOT","FR_FOOT","FL_FOOT"]
    
    
    rgeomModel.collision_pairs = []
    for name in frames_names:
        frame_id = rmodel.getFrameId(name)
        frame = rmodel.frames[frame_id]
        joint_id = frame.parent
        frame_placement = frame.placement
        
        shape_name = name + "_shape"
        shape = Ellipsoid(joint_id, shape_name , r, frame_placement)
        geometry = pin.GeometryObject(shape_name, joint_id, shape, frame_placement)
        geometry.meshColor = np.array([1.0, 0.2, 0.2, 1.])
        
        geom_id = rgeomModel.addGeometryObject(geometry)
        
        foot_plane = CollisionPairPlaneEllipsoid(planeId, geom_id)
        rgeomModel.collision_pairs += [foot_plane]
        rgeomModel.computeColPairDist.append(False)
    rgeom_data = rgeomModel.createData()
    ddl = np.array([i for i in range(6,rmodel.nv)])
    return rmodel, rgeomModel, robot.visual_model, rdata, rgeom_data, ddl

rmodel, rgeom_model, rvisual_model, rdata, rgeom_data, _ = create_solo_model()
nq = rmodel.nq
nv = rmodel.nv
ROT_NULL = np.eye(3)

if False:
    vizer = init_viewer_ellipsoids(rmodel, rgeom_model,rvisual_model, open=True)
    q_init = 1*rmodel.referenceConfigurations["standing"]
    vizer.display(q_init)
    input("Displaying q_init, press enter to visualize target configuration")
    q_target = 1*rmodel.referenceConfigurations["standing"]
    q_target[2] += .1
    vizer.display(q_target)
    input()
    exit()

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

class DiffSimDyamicsModel(ExplicitDynamicsModel):

    def __init__(self, space, model, actuation, geom_model, coeff_friction, coeff_rest, nu: int, dt: float, N_samples : int = 1, noise_intensity: float = 0.) -> None:
        super().__init__(space, nu)
        self.act = actuation
        self.dt = dt
        self.sim = SimulatorNode(Simulator(model, geom_model, dt, coeff_friction, coeff_rest, dt_collision=dt, eps_contact=1e-3))
        self.N_samples = N_samples
        self.noise_intensity = noise_intensity

    def forward(self, x, u, data):
        data.x_tens_ = torch.tensor(x, requires_grad=True)
        data.u_tens_ = torch.tensor(u, requires_grad=True)
        act_tens = torch.from_numpy(self.act)
        data.xnext_tens_ = torch.zeros_like(data.x_tens_)
        for i in range(self.N_samples):
            u_noisy = data.u_tens_+ data.u_noise[i]
            # print(f"data.x_tens_.shape {data.x_tens_.shape}")
            # print(f"u_noisy.shape {u_noisy.shape}")
            data.xnext_tens_ += self.sim.makeStep(data.x_tens_, act_tens @ u_noisy, calcPinDiff=True)
        data.xnext_tens_ /= self.N_samples
        data.xnext[:] = data.xnext_tens_.detach().numpy()
        return
    
    def dForward(self, x_, u_, data):
        for i in range(self.space.ndx):
            grads = torch.autograd.grad(data.xnext_tens_[i], [data.x_tens_,data.u_tens_],retain_graph=True)
            data.Jx[i,:]  = grads[0].detach().numpy()
            data.Ju[i,:] = grads[1].detach().numpy()
        return
    
    def computeJacobians(self, x, u, y, data):
        print("calling computeJacobians custom")
        data.u_noise = self.noise_intensity*torch.randn((self.N_samples,)+data.u_tens_.size())
        ExplicitDynamicsData.computeJacobians(x,u,y,data)

    def createData(self) -> ExplicitDynamicsData:
        data = ExplicitDynamicsData(self.space.ndx, self.nu, self.nx2, self.ndx2)
        data.u_tens_ = torch.zeros(self.nu)
        data.u_noise = torch.zeros((self.N_samples,)+data.u_tens_.size())
        # shape_xnext = data.xnext.shape
        # data.x_tens_ = torch.zeros(shape_xnext)
        # data.xnext_tens_ = torch.zeros(shape_xnext)
        return data



def main(args: Args):
    import meshcat

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

    nx = 2*rmodel.nv
    space = manifolds.VectorSpace(nx)

    nu = nv - 6
    act_matrix = np.eye(nv, nu, -6)

    coeff_friction = 0.9
    coeff_rest = 0.0
    # dt = 0.01
    # Tf = 1.5
    dt = 5e-3
    Tf = 0.5
    nsteps = int(Tf / dt)
    print("nsteps: {:d}".format(nsteps))

    dynmodel = DiffSimDyamicsModel(space, rmodel, act_matrix, rgeom_model, coeff_friction, coeff_rest,nu, dt)

    q0 = 1*rmodel.qref
    dq0 = pin.difference(rmodel, rmodel.qref, q0)
    x0 = np.concatenate([dq0, np.zeros(nv)])

    tau = pin.rnea(rmodel, rdata, q0, np.zeros(nv), np.zeros(nv))
    u0, _, _, _ = np.linalg.lstsq(act_matrix, tau)
    u0 = np.zeros(nu)

    def constraint_quasistatic_torque(nodes, x0, u0, S):
        B = torch.zeros((nv, nv))  # B is the actuation matrix when u is control including FF
        B[6:, 6:] = torch.eye(nv - 6)
        def compute_static_torque(node, x, u, S):
            print(f"x: {x}, u: {u}")
            x_next = node.makeStep(torch.tensor(x), torch.tensor(S @ u), calcPinDiff=True)
            model = node.rmodel
            data = model.createData()
            q = node.rdata.q
            print(f"q: {q}")

            u_static = pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv))  # len(u_static) = 14
            Jn = node.rdata.Jn_
            Jt = node.rdata.Jt_
            A = (torch.vstack((B, Jn, Jt)).T).detach().numpy()
            tau_lamN_lamT = np.linalg.pinv(A) @ u_static

            tau_static = tau_lamN_lamT[:model.nv]
            tau_static_ = torch.tensor(tau_static)
            lamN_static = tau_lamN_lamT[model.nv:model.nv+4]
            lamT_static = tau_lamN_lamT[model.nv+4:]

            print(f"tau_static: {tau_static}")
            # print(f"lamN_static: {lamN_static}")
            # print(f"lamT_static: {lamT_static}")

            return tau_static_[6:].detach().numpy(), x_next.detach().numpy()

        u_list, x_list = [], [x0]
        first_node = nodes[0]
        print(f"first_node")
        u, _ = compute_static_torque(first_node, x0, u0, S)

        x = 1*x0
        print("other nodes")
        for node in nodes:
            u, x = compute_static_torque(node, x, u, S)
            u_list.append(1*u)
            x_list.append(1*x)
        return u_list, x_list

    sim_nodes = [SimulatorNode(Simulator(rmodel, rgeom_model, dt, coeff_friction, coeff_rest, dt_collision=dt)) for _ in range(nsteps)]

    # compute initial guess
    us_init, xs_init = constraint_quasistatic_torque(
        sim_nodes, x0, u0, act_matrix
    )

    # us_init = [u0] * nsteps
    # xs_init = [x0] * (nsteps + 1)

    x_tar1 = 1*x0
    # x_tar1[2] += 0.1  # jump up
    # add_objective_vis_models(x_tar1)

    u_max = rmodel.effortLimit[-6:]
    u_min = -1*u_max

    times = np.linspace(0, Tf, nsteps + 1)

    def get_task_schedule():
        weights1 = np.zeros(space.ndx)
        weights1[:3] = 4.0
        weights1[3:6] = 1e-2
        weights1[nv:] = 1e-3

        def weight_target_selector(i):
            x_tar = x_tar1
            weights = weights1
            return weights, x_tar

        return weight_target_selector

    task_schedule = get_task_schedule()

    def setup():
        w_u = np.eye(nu) * 1e-2
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

    problem = setup()

    vizer = init_viewer_ellipsoids(rmodel, rgeom_model,rvisual_model, open=True)
    vizer.display(pin.integrate(rmodel,rmodel.qref,x0[:nv]))

    tol = 1e-3
    verbose = proxddp.VerboseLevel.VERBOSE
    history_cb = proxddp.HistoryCallback()
    solver = proxddp.SolverFDDP(tol, verbose=verbose)
    if args.proxddp:
        mu_init = 1e-1
        rho_init = 0.0
        solver = proxddp.SolverProxDDP(tol, mu_init, rho_init, verbose=verbose)
    solver.max_iters = 200
    solver.registerCallback("his", history_cb)
    solver.setup(problem)
    solver.run(problem, xs_init, us_init)

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
