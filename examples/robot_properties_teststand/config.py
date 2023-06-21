"""
@file config.py
@author Maximilien Naveau (maximilien.naveau@gmail.com)
@license License BSD-3-Clause
@copyright Copyright (c) 2019, New York University and Max Planck Gesellschaft.
@date 2019-05-22
@brief Define the interface between the control and the hardware
"""

import numpy as np
import pinocchio as pin

from os import path
from math import pi
from pinocchio.utils import zero
from pinocchio.robot_wrapper import RobotWrapper
from diffsim.shapes import Plane, Ellipsoid
from diffsim.collision_pairs import CollisionPairPlaneEllipsoid
from hppfcl import Halfspace, Sphere

def find_paths(robot_name):
    package_dir = path.dirname(path.abspath(__file__))
    # resources_dir = path.join(package_dir, "src")
    urdf_path = path.join(package_dir, "pre_generated_urdf" ,f"{robot_name}.urdf")

    paths = {"package":str(package_dir),
             "urdf":str(urdf_path)}

    return paths

class TeststandConfig:
    # name that is used by every other entities.
    robot_family = "teststand"
    robot_name = "teststand"

    # PID gains
    kp = 5.0
    kd = 0.1
    ki = 0.0

    # Here we use the same urdf as for the quadruped but without the freeflyer.
    paths = find_paths(robot_name)
    meshes_path = paths["package"]
    urdf_path = paths["urdf"]

    # The inertia of a single blmc_motor
    motor_inertia = 0.0000045

    # The motor gear ratio
    motor_gear_ration = 9.

    # The Kt constant of the motor [Nm/A]: tau = I * Kt
    motor_torque_constant = 0.025

    # pinocchio model
    robot_model = pin.buildModelFromUrdf(urdf_path)
    robot_model.rotorInertia[1:] = motor_inertia
    robot_model.rotorGearRatio[1:] = motor_gear_ration

    # the number of motors, here they are the same as there are only revolute
    # joints
    nb_joints = robot_model.nv - 1

    # control time period
    control_period = 0.001
    dt = control_period

    # maxCurrent = 12 # Ampers
    max_current = 2

    # maximum torques
    max_torque = motor_torque_constant * max_current

    # maximum control one can send, here the control is the current.
    max_control = max_current

    # mapping between the ctrl vector in the device and the urdf indexes
    urdf_to_dgm = (0, 1, 2)

    # ctrl_manager_current_to_control_gain I am not sure what it does so 1.0.
    ctrl_manager_current_to_control_gain = 1.0

    map_joint_name_to_id = {}
    map_joint_limits = {}
    for i, (name, lb, ub) in enumerate(zip(robot_model.names[1:],
                                           robot_model.lowerPositionLimit,
                                           robot_model.upperPositionLimit)):
        map_joint_name_to_id[name] = i
        map_joint_limits[i] = [float(lb), float(ub)]

    max_qref = pi

    # Define the initial state.
    # initial_configuration = [0.4, 0.8, -1.6]
    initial_configuration = [0.235, 0.8, -1.6]
    initial_velocity = 3*[0.0, ]

    q0 = zero(robot_model.nq)
    q0[:] = initial_configuration
    v0 = zero(robot_model.nv)
    a0 = zero(robot_model.nv)

    @classmethod
    def buildRobotWrapper(cls):
        # Rebuild the robot wrapper instead of using the existing model to
        # also load the visuals.
        robot = RobotWrapper.BuildFromURDF(cls.urdf_path, cls.meshes_path)
        robot.model.rotorInertia[1:] = cls.motor_inertia
        robot.model.rotorGearRatio[1:] = cls.motor_gear_ration
        return robot
    
    # @classmethod
    # def create_solo_leg_model(cls):
    #     robot = cls.buildRobotWrapper()
    #     rmodel = robot.model.copy()

    #     rmodel.qref  = np.array([0.235, 0.8, -1.6])
    #     rmodel.qinit = np.array([0.235, 0.8, -1.6])
    #     # Geometry model
    #     geom_model = robot.collision_model
    #     # add feet
    #     a = 0.02
    #     r = np.array([a, a, a])

    #     geom_model.computeColPairDist = []

    #     n = np.array([0., 0., 1])
    #     p = np.array([0., 0., 0.0])
    #     h = np.array([100., 100., 0.01])
    #     plane_shape = Plane(0, 'plane', n, p, h)
    #     T = pin.SE3(plane_shape.R, plane_shape.t)
    #     plane = pin.GeometryObject("plane", 0, 0, plane_shape, T)
    #     plane.meshColor = np.array([0.5, 0.5, 0.5, 1.]) 
    #     planeId = geom_model.addGeometryObject(plane)
        
    #     frames_names = ["contact"]
            
    #     geom_model.collision_pairs = []
    #     for name in frames_names:
    #         frame_id = rmodel.getFrameId(name)
    #         frame = rmodel.frames[frame_id]
    #         joint_id = frame.parent
    #         frame_placement = frame.placement
    #         # frame_placement.translation += np.array([0., 0., a])
            
    #         shape_name = name + "_shape"
    #         shape = Ellipsoid(joint_id, shape_name , r, frame_placement)
    #         geometry = pin.GeometryObject(shape_name, joint_id, shape, frame_placement)
    #         geometry.meshColor = np.array([1.0, 0.2, 0.2, .5])
            
    #         geom_id = geom_model.addGeometryObject(geometry)
            
    #         foot_plane = CollisionPairPlaneEllipsoid(planeId, geom_id)
    #         geom_model.collision_pairs += [foot_plane]
    #         geom_model.computeColPairDist.append(False)

    #     return rmodel, geom_model, robot.visual_model
    @classmethod
    def create_solo_leg_model(cls):
        robot = cls.buildRobotWrapper()
        rmodel = robot.model.copy()

        rmodel.qref  = np.array([0.235, 0.8, -1.6])
        rmodel.qinit = np.array([0.235, 0.8, -1.6])
        # Geometry model
        geom_model = robot.collision_model
        geom_model_cb = geom_model.copy()
        # add feet
        a = 0.02
        r = np.array([a, a, a])

        geom_model.computeColPairDist = []

        n = np.array([0., 0., 1])
        p = np.array([0., 0., 0.0])
        h = np.array([100., 100., 0.01])
        plane_shape = Plane(0, 'plane', n, p, h)
        T = pin.SE3(plane_shape.R, plane_shape.t)
        plane = pin.GeometryObject("plane", 0, 0, plane_shape, T)
        plane.meshColor = np.array([0.5, 0.5, 0.5, 1.]) 
        planeId = geom_model.addGeometryObject(plane)
        
        frames_names = ["contact"]
            
        geom_model.collision_pairs = []
        for name in frames_names:
            frame_id = rmodel.getFrameId(name)
            frame = rmodel.frames[frame_id]
            joint_id = frame.parent
            frame_placement = frame.placement
            
            shape_name = name + "_shape"
            shape = Ellipsoid(joint_id, shape_name , r, frame_placement)
            geometry = pin.GeometryObject(shape_name, joint_id, shape, frame_placement)
            geometry.meshColor = np.array([1.0, 0.2, 0.2, 1.])
            
            geom_id = geom_model.addGeometryObject(geometry)
            
            foot_plane = CollisionPairPlaneEllipsoid(planeId, geom_id)
            geom_model.collision_pairs += [foot_plane]
            geom_model.computeColPairDist.append(False)

            ##################################
            n = np.array([0.0, 0.0, 1])
            p = np.array([0.0, 0.0, 0.0])
            h = np.array([100.0, 100.0, 0.01])
            plane_shape = Halfspace(n, 0)
            # plane_shape = Halfspace(n, 0)
            T = pin.SE3(np.eye(3), np.zeros(3))
            ground_go = pin.GeometryObject("plane", 0, 0, T, plane_shape)
            ground_go.meshColor = np.array([0.5, 0.5, 0.5, 1.0])

            ground_id = geom_model_cb.addGeometryObject(ground_go)

            geom_model_cb.removeAllCollisionPairs()
            geom_model_cb.frictions = []
            geom_model_cb.elasticities = []
            frame_id = rmodel.getFrameId(name)
            frame = rmodel.frames[frame_id]
            joint_id = frame.parentJoint
            frame_placement = frame.placement

            shape_name = name + "_shape"
            shape = Sphere(a)
            geometry = pin.GeometryObject(shape_name, joint_id, frame_placement, shape)
            geometry.meshColor = np.array([1.0, 0.2, 0.2, 1.0])

            geom_id = geom_model_cb.addGeometryObject(geometry)

            foot_plane = pin.CollisionPair(ground_id, geom_id)  # order should be inverted ?
            geom_model_cb.addCollisionPair(foot_plane)
            mu, el = 0.7, 0.
            geom_model_cb.frictions += [mu]
            geom_model_cb.elasticities += [el]

        return rmodel, (geom_model, geom_model_cb), robot.visual_model