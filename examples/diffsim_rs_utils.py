import pinocchio as pin
import example_robot_data as erd
import numpy as np
import torch
import proxddp
import matplotlib.pyplot as plt
import proxsuite
import math

from diffsim.simulator import Simulator, SimulatorNode
from diffsim.shapes import Plane, Ellipsoid
from diffsim.collision_pairs import CollisionPairPlaneEllipsoid

from proxddp.dynamics import ExplicitDynamicsModel, ExplicitDynamicsData

from pycontact.simulators import NCPPGSSimulator, CCPADMMSimulator, NCPStagProjSimulator, LCPQPSimulator

def constraint_quasistatic_torque_contact_bench(model, geom_model, x0, St, T, dt, K, version = "lstsq"):
    # simulator = NCPPGSSimulator()
    # simulator = CCPADMMSimulator()
    # simulator = NCPStagProjSimulator()
    simulator = LCPQPSimulator()

    nv = model.nv
    nq = model.nq
    nu = St.shape[1]
    
    B = np.zeros((nv, nv))  # B is the actuation matrix when u is control including z_joint
    B[1:, 1:] = np.eye(nu)
    
    def compute_static_torque(x, St, dt, K, e_prev):
        u = np.zeros(nu)
        data = model.createData()
        geom_data = geom_model.createData()

        q = x[:model.nq]
        v = x[model.nq:]
        fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
        simulator.step(
            model, data, geom_model, geom_data, q, v, St @ u, fext, dt, K, 1, th=1e-10
        )
        J = simulator.J
        n_contact_points = len(simulator.contact_points)
        assert n_contact_points == 1

        if version == "lstsq":
            u_static = pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv))
            A = (np.vstack((B, J))).T
            tau_lamT_lamN = np.linalg.lstsq(A, u_static)[0]

            ddq = np.zeros(nv)
            tau_static = tau_lamT_lamN[1:model.nv]
            # lamT_static = tau_lamT_lamN[model.nv:model.nv+2]
            # lamN_static = tau_lamT_lamN[model.nv+2:]
            # print(f"q: {q}")
            # print(f"tau_static: {tau_static}")
            # print(f"lamN_static: {lamN_static}")
            # print(f"lamT_static: {lamT_static}")

        elif version == "qp":
            Rc = simulator.R
            Tc = simulator.contact_points[0]
            Kp = 1.*1e-1
            Kd = 1.*2*math.sqrt(Kp)
            e = 1*simulator.e  # if you are not on hppfcl this here is -e
            if e_prev is None:
                de = np.zeros(3)
            else:
                de = (e - e_prev) / dt
            # print(f"[LOCAL] e: {e}")
            # print(f"[LOCAL] de: {de}")

            oMc = pin.SE3(Rc, Tc)
            pin.framesForwardKinematics(model, data, q)
            oMe = data.oMf[-1]
            pMe = model.frames[-1].placement
            eMc = oMe.actInv(oMc)
            updated_pMe = pMe.act(eMc)

            # print(f"oMe: {oMe}")
            # print(f"oMc: {oMc}")
            # print(f"eMc: {eMc}")
            # print(f"pMe: {pMe}")
            # print(f"up_pMe: {updated_pMe}")
            # check_oMe = data.oMi[-1].act(updated_pMe)
            # print(f"check_oMe: {check_oMe}")

            model.frames[-1].placement = updated_pMe
            pin.crba(model, data, q)
            M = data.M
            c_plus_g = pin.rnea(model, data, q, v, np.zeros(model.nv))
            pin.forwardKinematics(model, data, q, v, np.zeros(model.nv))  # this will update the contact frame acceleration drift with 0 acceleration
            contact_acc = pin.getFrameClassicalAcceleration(model, data, model.getFrameId("contact"), pin.ReferenceFrame.LOCAL)
            # print(f"[LOCAL]\t\t\t contact_acc: {contact_acc.linear}")
            contact_acc_print = pin.getFrameClassicalAcceleration(model, data, model.getFrameId("contact"), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            # print(f"[LOCAL_WORLD_ALIGNED]\t contact_acc: {contact_acc_print.linear}")


            # construct the QP problem
            # min_{ddq, tau, lam} 0.5 || ddq ||^2_2
            # s.t. M*ddq + C + G = S^T @ tau + J^T @ lam
            #      J*ddq + acc_drift = acc_desired
            # x is of size 2*nv + nu

            H = np.zeros((2*nv + nu, 2*nv + nu))
            H[:nv, :nv] = np.eye(nv)
            g = None

            A_cstr1 = np.hstack((M, -St, -J.T))
            b_cstr1 = - c_plus_g 
            
            A_cstr2 = np.hstack((J, np.zeros((nv, nv+nu))))
            b_cstr2 = contact_acc.linear - Kp * e - Kd * de

            A_cstr = np.vstack((A_cstr1, A_cstr2))
            b_cstr = np.hstack((b_cstr1, b_cstr2))
            C = None
            l = None
            u = None

            n = 2*nv + nu
            n_eq = A_cstr.shape[0]
            n_in = 0

            qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
            qp.init(H, g, A_cstr, b_cstr, C, l, u)
            qp.settings.eps_abs = 1e-9
            qp.settings.max_iter = 10_000
            qp.solve()
            # print(f"status: {qp.results.info.status}, dual: {qp.results.info.dua_res}, primal: {qp.results.info.pri_res}")

            ddq = qp.results.x[:nv]
            tau_static = qp.results.x[nv:nv+nu]
            lam = qp.results.x[nv+nu:]

            lhs_qp_cstr1 = M @ ddq + c_plus_g
            rhs_qp_cstr1 = St @ tau_static + J.T @ lam

            lhs_qp_cstr2 = J @ ddq - contact_acc.linear
            rhs_qp_cstr2 = - Kp * e - Kd * de

            # print(f"[QP] constraint violation 1: {lhs_qp_cstr1 - rhs_qp_cstr1}")
            # print(f"[QP] constraint violation 2: {lhs_qp_cstr2 - rhs_qp_cstr2}")

            # print(f"[QP] ddq: {ddq}")
            # print(f"[QP] tau: {tau_static}")
            # print(f"[QP] lam: {lam}")

        return tau_static, e, contact_acc_print.linear, ddq
    
    def apply_static_torque(x, u, S, dt, K):
        data = model.createData()
        geom_data = geom_model.createData()

        q  = x[:model.nq]
        v = x[model.nq:]
        # print(f"q: {q}, v: {v}, u: {u}")
        fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
        q_next, v_next = simulator.step(
            model, data, geom_model, geom_data, q, v, S @ u, fext, dt, K, int(1e6), th=1e-10
        )
        # print(f"stopping: {simulator.solver.stop_}")
        
        q = q_next.copy()
        v = v_next.copy()
        # print(f"q: {q}, v: {v}")
        x_next = np.concatenate([q,v])
        return x_next

    q0 = pin.integrate(model, model.qref, x0[:nv])
    x = np.concatenate([q0, x0[nv:2*nv]])
    u_list, x_list, v_list, acc_c_list, ddq_list = [], [x], [], [], []
    e = None
    for i in range(T):
        print(f"[{i}] ----------------------------------------")
        u, e, acc_c, ddq = compute_static_torque(x, St, dt, K, e)
        u_list.append(1*u)
        x = apply_static_torque(x, u, St, dt, K)
        x_list.append(1*x)
        v_list.append(x[nq:nv+nq])
        acc_c_list.append(acc_c)
        ddq_list.append(ddq)

    if 0:
        from pinocchio.visualize import MeshcatVisualizer
        from time import sleep
        vizer = MeshcatVisualizer(model, geom_model, geom_model)
        vizer.initViewer(loadModel=True, open=True)
        vizer.display(x_list[0][:nq])
        sleep(1)
        for q in x_list:
            vizer.display(q[:nq])
            print(f"q: {q[:nq]}")
            sleep(0.1)
    if 0:
        plt.figure()
        plt.plot([x[:3] for x in x_list])
        plt.legend(["q1", "q2", "q3"])
        plt.figure()
        plt.plot(v_list)
        plt.legend(["v1", "v2", "v3"])
        plt.figure()
        plt.plot(acc_c_list)
        plt.legend(["acc_c1", "acc_c2", "acc_c3"])
        plt.figure()
        plt.plot(ddq_list)
        plt.legend(["ddq1", "ddq2", "ddq3"])
        plt.show()

    # need to convert x = [q,v] again back to x = [dq, v]
    x_list = [np.concatenate((pin.difference(model, model.qref, x[:nq]), x[nq:2*nv])) for x in x_list]
    return u_list, x_list

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

def constraint_quasistatic_torque_diffsim(nodes, x0, St, version = "lstsq"):
    nv = nodes[0].rmodel.nv
    nu = St.shape[1]
    dt = nodes[0].sim.dt

    B = torch.zeros((nv, nv))  # B is the actuation matrix when u is control including z_joint
    B[1:, 1:] = torch.eye(nv - 1)
    
    def compute_static_torque(node, x, St, e_prev):
        u = np.zeros(St.shape[1])

        model = node.rmodel
        data = model.createData()

        q = pin.integrate(model, model.qref, x[:model.nv])
        v = x[model.nv:]

        node.makeStep(torch.tensor(x), torch.tensor(St @ u), calcPinDiff=True)
        e = contact_acc_print = ddq = None
        if node.collisionChecker.ncol == 1:
            in_collision = True
            J = node.rdata.J_

        else:
            in_collision = False

        if version == "lstsq":

            u_static = pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv))
            A = (torch.vstack((B, J)).T).detach().numpy()
            tau_lamT_lamN = np.linalg.lstsq(A, u_static)[0]

            tau_static = tau_lamT_lamN[1:model.nv]
            # lamT_static = tau_lamT_lamN[model.nv:model.nv+1]
            # lamN_static = tau_lamT_lamN[model.nv+1:]
            # print(f"tau_static: {tau_static}")
            # print(f"lamN_static: {lamN_static}")
            # print(f"lamT_static: {lamT_static}")

        elif version == "qp":
            
            if in_collision:
                J = J.detach().numpy()
                # Rc = simulator.R  in diffsim oRc_: 
                # tensor([[-1.,  0.,  0.],
                        # [ 0.,  1.,  0.],
                        # [ 0.,  0., -1.]])
                # Rc = np.eye(3)
                Rc = node.collisionChecker.oRc
                Tc = node.collisionChecker.oTc
                Kp = 0.*1e-1
                Kd = 0.*2*math.sqrt(Kp)
                if node.e is None:
                    e = np.zeros(3)
                else:
                    e = np.array([0., 0., node.e.detach().numpy()[0]])
                if e_prev is None:
                    de = np.zeros(3)
                else:
                    de = (e - e_prev) / dt
                # print(f"[LOCAL] e: {e}")
                # print(f"[LOCAL] de: {de}")

                oMc = pin.SE3(Rc, Tc)
                pin.framesForwardKinematics(model, data, q)
                oMe = data.oMf[-1]
                pMe = model.frames[-1].placement
                eMc = oMe.actInv(oMc)
                updated_pMe = pMe.act(eMc)

                # print(f"oMe: {oMe}")
                # print(f"oMc: {oMc}")
                # print(f"eMc: {eMc}")
                # print(f"pMe: {pMe}")
                # print(f"up_pMe: {updated_pMe}")
                # check_oMe = data.oMi[-1].act(updated_pMe)
                # print(f"check_oMe: {check_oMe}")

                model.frames[-1].placement = updated_pMe
                pin.forwardKinematics(model, data, q, v, np.zeros(model.nv))  # this will update the contact frame acceleration drift with 0 acceleration
                contact_acc = pin.getFrameClassicalAcceleration(model, data, model.getFrameId("contact"), pin.ReferenceFrame.LOCAL)
                # print(f"[LOCAL]\t\t\t contact_acc: {contact_acc.linear}")
                contact_acc_print = pin.getFrameClassicalAcceleration(model, data, model.getFrameId("contact"), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                # print(f"[LOCAL_WORLD_ALIGNED]\t contact_acc: {contact_acc_print.linear}")

            pin.crba(model, data, q)
            M = data.M
            c_plus_g = pin.rnea(model, data, q, v, np.zeros(model.nv))

            # construct the QP problem
            # min_{ddq, tau, lam} 0.5 || ddq ||^2_2
            # s.t. M*ddq + C + G = S^T @ tau + J^T @ lam
            #      J*ddq + acc_drift = acc_desired
            # x is of size 2*nv + nu

            H = np.zeros((2*nv + nu, 2*nv + nu))
            H[:nv, :nv] = np.eye(nv)
            g = None

            b_cstr1 = - c_plus_g 
            if in_collision:
                A_cstr1 = np.hstack((M, -St, -J.T))
                A_cstr2 = np.hstack((J, np.zeros((nv, nv+nu))))
                b_cstr2 = contact_acc.linear - Kp * e - Kd * de
                A_cstr = np.vstack((A_cstr1, A_cstr2))
                b_cstr = np.hstack((b_cstr1, b_cstr2))
            else:
                A_cstr = np.hstack((M, -St, np.zeros((nv, nv))))
                b_cstr = b_cstr1

            C = None
            l = None
            u = None

            n = 2*nv + nu
            n_eq = A_cstr.shape[0]
            n_in = 0

            qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
            qp.init(H, g, A_cstr, b_cstr, C, l, u)
            qp.settings.eps_abs = 1e-9
            qp.settings.max_iter = 10_000
            qp.solve()
            # print(f"status: {qp.results.info.status}, dual: {qp.results.info.dua_res}, primal: {qp.results.info.pri_res}")

            ddq = qp.results.x[:nv]
            tau_static = qp.results.x[nv:nv+nu]
            lam = qp.results.x[nv+nu:]

            lhs_qp_cstr1 = M @ ddq + c_plus_g
            if in_collision:
                rhs_qp_cstr1 = St @ tau_static + J.T @ lam

                lhs_qp_cstr2 = J @ ddq - contact_acc.linear
                rhs_qp_cstr2 = - Kp * e - Kd * de
                # print(f"[QP] constraint violation 2: {lhs_qp_cstr2 - rhs_qp_cstr2}")

            else:
                rhs_qp_cstr1 = St @ tau_static
            # print(f"[QP] constraint violation 1: {lhs_qp_cstr1 - rhs_qp_cstr1}")

            # print(f"[QP] ddq: {ddq}")
            # print(f"[QP] tau: {tau_static}")
            # print(f"[QP] lam: {lam}")


        return tau_static, e, contact_acc_print, ddq

    u_list, x_list, v_list, acc_c_list, ddq_list = [], [x0], [], [], []
    x = 1*x0
    e = None
    for i, node in enumerate(nodes):
        print(f"[{i}] ----------------------------------------")
        u, e, acc_c, ddq = compute_static_torque(node, x, St, e)
        u_list.append(1*u)
        # print(f" q: {pin.integrate(node.rmodel, node.rmodel.qref, x[:nv])}, v: {x[nv:2*nv]}, u: {u}")
        x = node.makeStep(torch.tensor(x), torch.tensor(St @ u), calcPinDiff=True).detach().numpy()
        # print(f" q: {pin.integrate(node.rmodel, node.rmodel.qref, x[:nv])}, v: {x[nv:2*nv]}")
        x_list.append(1*x)
        v_list.append(x[nv:2*nv])
        acc_c_list.append(acc_c)
        ddq_list.append(ddq)

    model = nodes[0].rmodel
    # exit()
    if 0:
        from pinocchio.visualize import MeshcatVisualizer
        from time import sleep
        vizer = MeshcatVisualizer(model, nodes[0].rgeom_model, nodes[0].rgeom_model)
        vizer.initViewer(loadModel=True, open=True)
        vizer.display(pin.integrate(model, model.qref, x_list[0][:nv]))
        sleep(1)
        for q in x_list:
            q = pin.integrate(model, model.qref, q[:nv])
            vizer.display(q)
            print(f"q: {q}")
            sleep(0.1)
    if 0:
        plt.figure()
        plt.plot([x[:3] for x in x_list])
        plt.legend(["q1", "q2", "q3"])
        plt.figure()
        plt.plot(v_list)
        plt.legend(["v1", "v2", "v3"])
        plt.figure()
        # plt.plot(acc_c_list)
        # plt.legend(["acc_c1", "acc_c2", "acc_c3"])
        plt.figure()
        plt.plot(ddq_list)
        plt.legend(["ddq1", "ddq2", "ddq3"])
        plt.show()
    
    return u_list, x_list
