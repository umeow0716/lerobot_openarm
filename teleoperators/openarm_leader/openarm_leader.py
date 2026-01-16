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

import logging
import time
import openarm_can as oa

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
    name = "openarm_leader"

    def __init__(self, config: OpenArmConfig):
        super().__init__(config)
        self.config = config
        self.right_arm = oa.OpenArm(self.config.right_port, self.config.enable_fd)
        self.left_arm  = oa.OpenArm(self.config.left_port,  self.config.enable_fd)

        self._is_connected = False

    @property
    def action_features(self) -> dict[str, type]:
        action = {}
        
        for i in range(8):
            action[f'RJ{i+1}.pos'] = float
            action[f'LJ{i+1}.pos'] = float
        
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

        if calibrate and not self.is_calibrated:
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

    def configure(self) -> None:
        self.right_arm.init_arm_motors(self.config.motor_types, self.config.send_ids, self.config.recv_ids)
        self.right_arm.init_gripper_motor(self.config.gripper_motor_type, self.config.gripper_motor_send_id, self.config.gripper_motor_recv_id)
        self.right_arm.get_arm().set_control_mode_all(oa.ControlMode.MIT) # type: ignore
        self.right_arm.get_gripper().set_control_mode_all(oa.ControlMode.MIT) # type: ignore
        self.right_arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.right_arm.enable_all()
        
        self.left_arm.init_arm_motors(self.config.motor_types, self.config.send_ids, self.config.recv_ids)
        self.left_arm.init_gripper_motor(self.config.gripper_motor_type, self.config.gripper_motor_send_id, self.config.gripper_motor_recv_id)
        self.left_arm.get_arm().set_control_mode_all(oa.ControlMode.MIT) # type: ignore
        self.left_arm.get_gripper().set_control_mode_all(oa.ControlMode.MIT) # type: ignore
        self.left_arm.set_callback_mode_all(oa.CallbackMode.STATE)
        self.left_arm.enable_all()

    def setup_motors(self) -> None:
        raise NotImplementedError('setup_motors() method not implemented in OpenArmLeader')

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        
        self.right_arm.refresh_all()
        self.right_arm.recv_all()
        
        self.left_arm.refresh_all()
        self.left_arm.recv_all()

        action = {}
        for i, motor in enumerate(self.right_arm.get_arm().get_motors()):
            action[f'RJ{i+1}.pos'] = motor.get_position()
        action['RJ8.pos'] = self.right_arm.get_gripper().get_motor().get_position()
        
        for i, motor in enumerate(self.left_arm.get_arm().get_motors()):
            action[f'LJ{i+1}.pos'] = motor.get_position()
        action['LJ8.pos'] = self.left_arm.get_gripper().get_motor().get_position()
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._is_connected = False
        time.sleep(0.1)
        self.right_arm.disable_all()
        self.left_arm.disable_all()
        self.right_arm.recv_all(610)
        self.left_arm.recv_all(610)

        logger.info(f"{self} disconnected.")