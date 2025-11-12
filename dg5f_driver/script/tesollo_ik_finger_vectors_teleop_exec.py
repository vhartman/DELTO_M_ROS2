#!/usr/bin/env -S uv run --script

from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

import zmq
import numpy as np
import threading 

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import time

import sys
import os

from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from keyboard_util import KeyboardThread, get_keyboard_flag, reset_keyboard_flag


_HERE = Path(__file__).parent
_XML = "/home/duplo/git/robohand/assets/output_mocap.xml"

class ManusSkeletonReceiver:
    '''
    Applies to any ZMQ-broadcasted mocap data with fixed shape (21,3) and dtype float32.
    Runs a background thread to continuously receive and update latest data.
    '''
    def __init__(self, port=8000, glove_serial_number="dc8b6ba8"):
        self.serial_number = glove_serial_number

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(
            zmq.CONFLATE, 1
        )  # Keep only the latest message
        self.socket.connect(f"tcp://localhost:{port}")

        self._latest_pos = None
        self._latest_quat = None

        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        self.fingertip_indices = [4, 9, 14, 19, 24]

    def _recv_loop(self):

        # ordered as thumb, index, middle, ring, little
        # chain_order = [
        #     0, 21, 22, 23, 24, 
        #     1, 2, 3, 4, 5,
        #     6, 7, 8, 9, 10,
        #     16, 17, 18, 19, 20,
        #     11, 12, 13, 14, 15,
        # ]
        chain_order = [
            0, 1, 2, 3, 4, 
            5, 6, 7, 8, 9,
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19,
            20, 21, 22, 23, 24,
        ]

        theta = np.pi  # 90 degrees
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])

        Ry = R.from_euler('y', -90, degrees=True).as_matrix()

        while self._running:
            try:
                message = self.socket.recv()
                message = message.decode('utf-8')
                data = message.split(",")

                assert len(data) == 176

                serial_number = data[0]
                assert serial_number == self.serial_number
                floats = np.array(data[1:], dtype=float).reshape(-1, 7)

                pos = floats[:, :3]
                quats = floats[:, 3:]

                # --- Remove wrist rotation ---
                R_wrist = R.from_quat(quats[0]).as_matrix()
                R_inv = R_wrist.T  # inverse rotation
                pos = (R_inv @ (pos - pos[0]).T).T  # unrotate & recenter at wrist
                rotmats = R_inv @ R.from_quat(quats).as_matrix()

                # --- Apply additional 90° rotation around world y-axis ---
                pos = (Ry @ pos.T).T
                rotmats = Ry @ rotmats
                quats = R.from_matrix(rotmats).as_quat()

                pos = pos[chain_order, :]
                quats = quats[chain_order, :]

                with self._lock:
                    self._latest_pos = pos
                    self._latest_quat = quats
            except zmq.Again:
                time.sleep(0.001)

    def get(self):
        with self._lock:
            if self._latest_pos is not None:
                return {"result": self._latest_pos.copy(), "status": "recording"}
            else:
                return {"result": None, "status": "no data"}
            
    def get_fingertip_pos(self):
        with self._lock:
            if self._latest_pos is not None:
                return self._latest_pos[self.fingertip_indices, :]
            else:
                return None
            
    def get_fingertip_quat(self):
        with self._lock:
            if self._latest_quat is not None:
                return self._latest_quat[self.fingertip_indices, :]
            else:
                return None


    def close(self):
        self._running = False
        self._thread.join()
        self.socket.close()



class MujocoRetargeting:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(_XML)

        self.mocap = ManusSkeletonReceiver()

        self.configuration = mink.Configuration(self.model)

        self.posture_task = mink.PostureTask(self.model, cost=1e-2)

        self.finger_pairs = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
        ]

        self.fingers = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]
        self.finger_tasks = []
        for finger in self.fingers:
            task = mink.FrameTask(
                frame_name=finger,
                frame_type="geom",
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=.5,
            )
            self.finger_tasks.append(task)

        self.relative_finger_tasks = []
        # for idx1, idx2 in self.finger_pairs:
        #     task = mink.RelativeFrameTask(
        #         frame_name=self.fingers[idx1],
        #         frame_type="geom",
        #         root_name=self.fingers[idx2],
        #         root_type="geom",
        #         position_cost=0.,
        #         orientation_cost=0.0,
        #         lm_damping=1.0,
        #     )
        #     self.relative_finger_tasks.append(task)

        collision_pairs = [
            (["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"], ["palm_lower", "palm_upper"]),
            (["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"], ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]),
            (["thumb_base", "index_base", "middle_base", "ring_base", "little_base"], ["thumb_base", "index_base", "middle_base", "ring_base", "little_base"]),
            (["little_tip"], ["ring_base", "ring_before_tip"]),
        ]

        self.limits = [
            mink.ConfigurationLimit(model=self.model),
            # mink.CollisionAvoidanceLimit(model=self.model, geom_pairs=collision_pairs),
        ]

        self.tasks = [
            self.posture_task,
            *self.finger_tasks,
            # *relative_finger_tasks
        ]

        self.model = self.configuration.model
        self.data = self.configuration.data
        self.solver = "daqp"

        self.scale = 1.25
        common_x_offset = 0.02
        common_y_offset = -0.00
        self.offsets = {"thumb_tip": np.array([common_x_offset, common_y_offset, 0]),
                "index_tip": np.array([common_x_offset, common_y_offset, 0]), 
                "middle_tip": np.array([common_x_offset, common_y_offset, 0]), 
                "ring_tip": np.array([common_x_offset, common_y_offset, 0]), 
                "little_tip": np.array([common_x_offset, common_y_offset, 0])}
        
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

        self._latest_joint_data = np.zeros(20)

    def run(self):
        with mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)

            # configuration.update_from_keyframe("grasp hard")

            # Initialize mocap bodies at their respective sites.
            self.posture_task.set_target_from_configuration(self.configuration)
            for finger in self.fingers:
                mink.move_mocap_to_frame(self.model, self.data, f"{finger}_target", finger, "geom")

            rate = RateLimiter(frequency=500.0, warn=False)
            dt = rate.dt
            t = 0
            while viewer.is_running() and self._running:
                # get task target from mocap
                fingertip_positions = self.mocap.get_fingertip_pos()
                if fingertip_positions is not None:
                    # set mocap objects to fingetips from manus
                    for i, finger in enumerate(self.fingers):
                        mocap_id = self.model.body(f"{finger}_target").mocapid[0]

                        pos = (fingertip_positions[i] + self.offsets[finger]) * self.scale
                        self.data.mocap_pos[mocap_id] = pos.copy()

                    # for (idx1, idx2), task in zip(finger_pairs, relative_finger_tasks):
                    #     translation = fingertip_positions[idx1] - fingertip_positions[idx2]
                    #     print(idx1, idx2, translation)
                    #     T = mink.SE3.from_translation(translation)
                    #     task.set_target(T)
                # else:
                #     for (idx1, idx2), task in zip(finger_pairs, relative_finger_tasks):
                #         T = mink.SE3.identity()
                #         task.set_target(T)

                # Update task target.
                for finger, task in zip(self.fingers, self.finger_tasks):
                    task.set_target(
                        mink.SE3.from_mocap_name(self.model, self.data, f"{finger}_target")
                    )

                vel = mink.solve_ik(self.configuration, self.tasks, rate.dt, self.solver, 1e-5, limits=self.limits)
                self.configuration.integrate_inplace(vel, rate.dt)
                mujoco.mj_camlight(self.model, self.data)

                with self._lock:
                    self._latest_joint_data = self.data.qpos.copy()

                # Visualize at fixed FPS.
                viewer.sync()
                rate.sleep()
                t += dt

    def get(self):
        with self._lock:
            return self._latest_joint_data.tolist()

        
    def close(self):
        self._running = False
        self._thread.join()        


class JointTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('joint_trajectory_publisher')
        self.publisher_ = self.create_publisher(
            JointTrajectory, '/dg5f_right_controller/joint_trajectory', 1)
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.loop)
        self.joint_names = ["rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",
                            "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
                            "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",
                            "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
                            "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4"]
        

        self.mj_ik = MujocoRetargeting()

    def loop(self):
        # update enter receiver
        if True or get_keyboard_flag():
            reset_keyboard_flag()

            message = JointTrajectory()
            message.joint_names = self.joint_names

            point = JointTrajectoryPoint()

            configuration = self.mj_ik.get()
            point.positions = configuration
            
            point.time_from_start = Duration(sec=0, nanosec=0)

            message.points.append(point)
            # self.get_logger().info("Joint Trajectory publish : {}".format(
            #     configuration))
            self.publisher_.publish(message)


def main(args=None):
    rclpy.init(args=args)
    joint_trajectory_publisher = JointTrajectoryPublisher()
    rclpy.spin(joint_trajectory_publisher)
    joint_trajectory_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    KeyboardThread()

    print("AAAA")

    main()
