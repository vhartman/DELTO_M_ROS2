#!/usr/bin/env python3

# Copyright 2025 tesollo
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the tesollo nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import zmq
import threading
import time
import numpy as np

import mujoco
import mujoco.viewer

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from keyboard_util import KeyboardThread, get_keyboard_flag, reset_keyboard_flag


def d2r(deg):
    return deg * 3.141592 / 180.0

def right_ergonomics_to_mujoco(ergonomics_data):
    ergonomics_data[1] = ergonomics_data[1] - 90.
    ergonomics_data[2] = ergonomics_data[2]

    for i in [1, 2, 3]:
        ergonomics_data[i * 4] = -ergonomics_data[i*4] + 0.

    tmp = ergonomics_data[0]
    ergonomics_data[0] = ergonomics_data[1]
    ergonomics_data[1] = tmp

    ergonomics_data[0] = -ergonomics_data[0]
    ergonomics_data[1] = -ergonomics_data[1] - 45

    tmp = ergonomics_data[16]
    ergonomics_data[16] = ergonomics_data[17] * 0.5
    ergonomics_data[17] = -tmp

    
    # ergonomics_data[16] = 0
    # ergonomics_data[17] = 0
    # ergonomics_data[18] = 0
    # ergonomics_data[19] = 0

    return ergonomics_data


class ManusErgonomicsReceiver():
    def __init__(self, socket_address="tcp://localhost:8001", update_rate = 1/100.):
        self.socket_address = socket_address

        self.context = None
        self.socket = None

        self.update_rate = update_rate
        self.pose = np.zeros(20)

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        self.disp = True

        if self.disp:
            model_path = "/home/valentin/git/postdoc/tesollo/output.xml"

            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.mj_data = mujoco.MjData(self.model)
            self.viewer = mujoco.viewer.launch_passive(self.model, self.mj_data)
                
    def start(self):
        """Start the receiver thread"""
        try:
            print("started")
            # Initialize ZMQ context and socket
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PULL)
            self.socket.setsockopt(
                zmq.CONFLATE, 1
            )  # Keep only the latest message
            self.socket.connect(self.socket_address)
            # self.socket.setsockopt(zmq.SUBSCRIBE, b"")

            self._running = True

            # Create thread and start it
            self._thread = threading.Thread(target=self._loop)
            self._thread.daemon = True
            self._thread.start()

            return True
        
        except Exception as e:
            print(f"Failed to start: {e}")
            if hasattr(self, "socket"):
                self.socket.close()
            if hasattr(self, "context"):
                self.context.term()
            return False

    def stop(self):
        """Stop the board pose estimation thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        # Clean up ZMQ resources
        if hasattr(self, "socket"):
            self.socket.close()
        if hasattr(self, "context"):
            self.context.term()

        if self.disp and hasattr(self, "viewer"):
            self.viewer.close()

    def _loop(self):
        try:
            while self._running:
                # Get box pose estimation
                try:
                    #wait for message
                    message = self.socket.recv()
                    #receive the message from the socket
                    message = message.decode('utf-8')
                    # print("Received reply %s" % (message))
                    data = message.split(",") 
                    if len(data) == 40:
                        print("updating")
                        with self._lock:
                            print("updating")

                            right_hand_ergonomics = list(map(float,data[20:40]))
                            right_hand_ergonomics = right_ergonomics_to_mujoco(right_hand_ergonomics)

                            self.pose = np.array(right_hand_ergonomics) / 360 * 2 * np.pi

                            if self.disp:
                                self.mj_data.qpos[:] = np.array(right_hand_ergonomics) / 360 * 2 * np.pi

                                mujoco.mj_forward(self.model, self.mj_data)

                                if self.viewer.is_running():
                                    self.viewer.sync()
                except zmq.Again:
                    pass

                # Sleep for a bit to avoid hogging CPU
                time.sleep(self.update_rate)
        except Exception as e:
            print(f"Error in board pose estimator thread: {e}")

        finally:
            # Clean up ZMQ socket
            self.socket.close()

    def get_angles_in_rad(self):
         with self._lock:
            return self.pose.tolist()


class JointTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('joint_trajectory_publisher')
        self.publisher_ = self.create_publisher(
            JointTrajectory, '/dg5f_right_controller/joint_trajectory', 10)
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.loop)
        self.joint_names = ["rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",
                            "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
                            "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",
                            "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
                            "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4"]
        

        self.manus_receiver = ManusErgonomicsReceiver()
        self.manus_receiver.start()

    def loop(self):
        # update enter receiver
        if True or get_keyboard_flag():
            reset_keyboard_flag()

            message = JointTrajectory()
            message.joint_names = self.joint_names

            point = JointTrajectoryPoint()

            configuration = self.manus_receiver.get_angles_in_rad()
            point.positions = configuration
            
            point.time_from_start = Duration(sec=0, nanosec=0)

            message.points.append(point)
            self.get_logger().info("Joint Trajectory publish : {}".format(
                configuration))
            self.publisher_.publish(message)


def main(args=None):
    rclpy.init(args=args)
    joint_trajectory_publisher = JointTrajectoryPublisher()
    rclpy.spin(joint_trajectory_publisher)
    joint_trajectory_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    KeyboardThread()

    main()
