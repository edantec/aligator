import pinocchio as pin
import example_robot_data as erd
import numpy as np
import torch
import proxddp

from diffsim.simulator import Simulator, SimulatorNode
from diffsim.shapes import Plane, Ellipsoid
from diffsim.collision_pairs import CollisionPairPlaneEllipsoid

from proxddp.dynamics import ExplicitDynamicsModel, ExplicitDynamicsData


def create_quadrotor_model():
    # robot model
    robot = erd.load("hector")
    rmodel = robot.model.copy()
    rdata = rmodel.createData()
    rmodel.qref = pin.neutral(rmodel)
    # rmodel.qref[2] = 1.87707555e-01 - 0.01
    

    # Geometry model
    rgeom_model = robot.collision_model
    rvisual_model = robot.visual_model

    a = 0.02

    r = np.array([a, a, a])  

    rgeom_model.computeColPairDist = []

    n = np.array([0., 0., 1])
    p = np.array([0., 0., 0.0])
    h = np.array([100., 100., 0.01])
    plane_shape = Plane(0, 'plane', n, p, h)
    T = pin.SE3(plane_shape.R, plane_shape.t)
    plane = pin.GeometryObject("plane", 0, 0, plane_shape, T)
    plane.meshColor = np.array([0.5, 0.5, 0.5, 1.]) 
    planeId = rgeom_model.addGeometryObject(plane)

    names = ["FR", "FL", "BR", "BL"]
    placements = [pin.SE3(np.eye(3), np.array([0.1,0.1,-0.168])), pin.SE3(np.eye(3), np.array([0.1,-0.1,-0.168])), pin.SE3(np.eye(3), np.array([-0.1,-0.1,-0.168])), pin.SE3(np.eye(3), np.array([-0.1,0.1,-0.168]))]
    sphere_radius = 0.1
    rgeom_model.collision_pairs = []
    for i,name in enumerate(names):
        # frame_id = rmodel.getFrameId("base_link")
        # frame = rmodel.frames[frame_id]
        joint_id = 1
        frame_placement = placements[i]
        shape_name = name + "_shape"
        # shape = fcl.Sphere(sphere_radius)
        shape = Ellipsoid(joint_id, "ellipsoid", r, frame_placement)
        geometry = pin.GeometryObject(shape_name, joint_id, frame_placement, shape)
        geometry.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        geom_id = rgeom_model.addGeometryObject(geometry)
        # foot_plane = pin.CollisionPair(ground_id, geom_id)  # order should be inverted ?
        # rgeom_model.addCollisionPair(foot_plane)

        cpFLeftFootPlane = CollisionPairPlaneEllipsoid(planeId, geom_id)
        rgeom_model.collision_pairs += [cpFLeftFootPlane]
        #rgeomModel.addCollisionPair(cpFLeftFootPlane)
        rgeom_model.computeColPairDist.append(False)
    # rgeom_model.collision_pairs = rgeom_model.collisionPairs.tolist()

    rgeom_data = rgeom_model.createData()

    # The matrix below maps rotor controls to torques, from proxddp 
    d_cog, cf, cm, u_lim, _ = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
    QUAD_ACT_MATRIX = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, d_cog, 0.0, -d_cog],
            [-d_cog, 0.0, d_cog, 0.0],
            [-cm / cf, cm / cf, -cm / cf, cm / cf],
        ]
    )
    
    return rmodel, rgeom_model, rvisual_model, rdata, rgeom_data, QUAD_ACT_MATRIX

def create_solo_model():
    robot = erd.load('solo12')
    rmodel = robot.model.copy()
    rdata = rmodel.createData()
    rmodel.qref = rmodel.referenceConfigurations["standing"]
    rmodel.qinit = rmodel.referenceConfigurations["standing"]
    # Geometry model
    rgeomModel = robot.collision_model
    # add feet
    a = 0.01910275
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


class DiffSimDynamicsModel(ExplicitDynamicsModel):

    def __init__(self, space, model, actuation, geom_model, coeff_friction, coeff_rest, nu: int, dt: float) -> None:
        super().__init__(space, nu)
        self.act = actuation
        self.dt = dt
        self.sim = SimulatorNode(Simulator(model, geom_model, dt, coeff_friction, coeff_rest, dt_collision=dt, eps_contact=1e-3))
        # self.N_samples = N_samples
        # self.noise_intensity = noise_intensity

    def forward(self, x, u, data):
        data.x_tens_ = torch.tensor(x, requires_grad=True)
        data.u_tens_ = torch.tensor(u, requires_grad=True)
        act_tens = torch.from_numpy(self.act)
        data.xnext_tens_ = torch.zeros_like(data.x_tens_)
        if data.resample_noise:
            data.u_noise = data.noise_intensity*torch.randn((data.N_samples,)+data.u_tens_.size())
            data.resample_noise = False
        for i in range(data.N_samples):
            u_noisy = data.u_tens_+ data.u_noise[i]
            data.xnext_tens_ += self.sim.makeStep(data.x_tens_, act_tens @ u_noisy, calcPinDiff=True)
        data.xnext_tens_ /= data.N_samples
        data.xnext[:2*self.sim.rmodel.nv] = data.xnext_tens_.detach().numpy()
        return
    
    def dForward(self, x_, u_, data):
        for i in range(2*self.sim.rmodel.nv):
            grads = torch.autograd.grad(data.xnext_tens_[i], [data.x_tens_,data.u_tens_],retain_graph=True)
            data.Jx[i,:2*self.sim.rmodel.nv]  = grads[0].detach().numpy()
            data.Ju[i,:2*self.sim.rmodel.nv] = grads[1].detach().numpy()
        return
    
    def createData(self) -> ExplicitDynamicsData:
        data = ExplicitDynamicsData(self.space.ndx, self.nu, self.nx2, self.ndx2)
        data.u_tens_ = torch.zeros(self.nu)
        data.resample_noise = False
        data.N_samples = 1
        data.noise_intensity = 0.0
        data.u_noise = torch.zeros((data.N_samples,)+data.u_tens_.size())
        # shape_xnext = data.xnext.shape
        # data.x_tens_ = torch.zeros(shape_xnext)
        # data.xnext_tens_ = torch.zeros(shape_xnext)
        return data
    
class DiffSimDynamicsModelAugmented(DiffSimDynamicsModel):
    def forward(self, x, u, data):
        super().forward(x[:(2*self.sim.rmodel.nv)], u, data)
        data.xnext[(2*self.sim.rmodel.nv):(2*self.sim.rmodel.nv+self.nu)] = u
        data.xnext[(2*self.sim.rmodel.nv+self.nu):] =  u - x[(2*self.sim.rmodel.nv):(2*self.sim.rmodel.nv+self.nu)]
        return

    def dForward(self, x_, u_, data):
        super().dForward(x_[:2*self.sim.rmodel.nv], u_, data)
        data.Jx[(2*self.sim.rmodel.nv+self.nu):,(2*self.sim.rmodel.nv):(2*self.sim.rmodel.nv+self.nu)] = -np.eye(self.nu)
        data.Ju[(2*self.sim.rmodel.nv):(2*self.sim.rmodel.nv+self.nu)] = np.eye(self.nu)
        data.Ju[(2*self.sim.rmodel.nv+self.nu):] = np.eye(self.nu)
        return
    
class RSCallback(proxddp.BaseCallback):
    def __init__(self, N_samples : int = 1, noise_intensity : float = 0.):
        super().__init__()
        self.N_samples = N_samples
        self.noise_intensity = noise_intensity

    def call(self, workspace: proxddp.Workspace, results: proxddp.Results):
        for sdi in workspace.problem_data.stage_data:
            for cdj in sdi.constraint_data:
                cdj.resample_noise = True
                cdj.N_samples = self.N_samples
                cdj.noise_intensity = self.noise_intensity


def constraint_quasistatic_torque(nodes, x0, u0, S):
    nv = nodes[0].rmodel.nv
    B = torch.zeros((nv, nv))  # B is the actuation matrix when u is control including z_joint
    B[1:, 1:] = torch.eye(nv - 1)
    def compute_static_torque(node, x, u, S):
        x_next = node.makeStep(torch.tensor(x), torch.tensor(S @ u), calcPinDiff=True)
        model = node.rmodel
        data = model.createData()
        q = node.rdata.q

        u_static = pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv))
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
    for node in nodes:
        u, x = compute_static_torque(node, x, u, S)
        u_list.append(1*u)
        x_list.append(1*x)
    return u_list, x_list