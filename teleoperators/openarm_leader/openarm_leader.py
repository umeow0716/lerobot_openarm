#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import logging
import time
import openarm_can as oa

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.teleoperators import Teleoperator
from .config_openarm_leader import OpenArmConfig


logger = logging.getLogger(__name__)


class OpenArmLeader(Teleoperator):
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
        expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    """

    config_class = OpenArmConfig
    name = "koch_leader"

    def __init__(self, config: OpenArmConfig):
        super().__init__(config)
        self.config = config
        self.right_arm = oa.OpenArm(self.config.right_port, self.config.enable_fd)
        self.left_arm  = oa.OpenArm(self.config.left_port,  self.config.enable_fd)
        
        self.right_refresh_thread = None
        self.left_refresh_thread = None
        
        self._is_connected = False

    @property
    def action_features(self) -> dict[str, type]:
        action = {}
        
        for i, motor in enumerate(self.right_arm.get_arm().get_motors()):
            action[f'RJ{i+1}.pos'] = motor.get_position()
        action['RJ8'] = self.right_arm.get_gripper().get_motor().get_position()
        
        for i, motor in enumerate(self.left_arm.get_arm().get_motors()):
            action[f'LJ{i+1}.pos'] = motor.get_position()
        action['LJ8'] = self.left_arm.get_gripper().get_motor().get_position()
        
        return action

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = False) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        self._is_connected = True
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        raise NotImplementedError('is_calibrated property not implemented in OpenArmLeader')

    def calibrate(self) -> None:
        raise NotImplementedError('calibrate() method not implemented in OpenArmLeader')

    # def configure(self) -> None:
    #     self.bus.disable_torque()
    #     self.bus.configure_motors()
    #     for motor in self.bus.motors:
    #         if motor != "gripper":
    #             # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
    #             # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while
    #             # assembling the arm, you could end up with a servo with a position 0 or 4095 at a crucial
    #             # point
    #             self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

    #     # Use 'position control current based' for gripper to be limited by the limit of the current.
    #     # For the follower gripper, it means it can grasp an object without forcing too much even tho,
    #     # its goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
    #     # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
    #     # to make it move, and it will move back to its original target position when we release the force.
    #     self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
    #     # Set gripper's goal pos in current position mode so that we can use it as a trigger.
    #     self.bus.enable_torque("gripper")
    #     if self.is_calibrated:
    #         self.bus.write("Goal_Position", "gripper", self.config.gripper_open_pos)
    
    def configure(self) -> None:
        self.right_arm.init_arm_motors(self.config.motor_types, self.config.send_ids, self.config.recv_ids)
        self.right_arm.init_gripper_motor(self.config.gripper_motor_type, self.config.gripper_motor_send_id, self.config.gripper_motor_recv_id)
        self.right_arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.right_arm.enable_all()
        self.right_refresh_thread = threading.Thread(target=self._refresh_thread, args=(self.right_arm, ), daemon=True)
        self.right_refresh_thread.start()
        
        self.left_arm.init_arm_motors(self.config.motor_types, self.config.send_ids, self.config.recv_ids)
        self.left_arm.init_gripper_motor(self.config.gripper_motor_type, self.config.gripper_motor_send_id, self.config.gripper_motor_recv_id)
        self.left_arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.left_arm.enable_all()
        self.left_refresh_thread = threading.Thread(target=self._refresh_thread, args=(self.left_arm, ), daemon=True)
        self.left_refresh_thread.start()

    def setup_motors(self) -> None:
        raise NotImplementedError('setup_motors() method not implemented in OpenArmLeader')

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        
        action = {}
        self.right_arm.refresh_all()
        for i, motor in enumerate(self.right_arm.get_arm().get_motors()):
            action[f'RJ{i+1}.pos'] = motor.get_position()
        action['RJ8'] = self.right_arm.get_gripper().get_motor().get_position()
        
        for i, motor in enumerate(self.left_arm.get_arm().get_motors()):
            action[f'LJ{i+1}.pos'] = motor.get_position()
        action['LJ8'] = self.left_arm.get_gripper().get_motor().get_position()
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.right_arm.disable_all()
        self.left_arm.disable_all()
        self._is_connected = False
        
        logger.info(f"{self} disconnected.")

    @staticmethod
    def _refresh_thread(arm: oa.OpenArm):
        while True:
            arm.refresh_all()
            arm.recv_all(610)