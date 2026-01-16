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

from dataclasses import dataclass, field

from openarm_can import MotorType
from lerobot.cameras.camera import CameraConfig

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("openarm_follower")
@dataclass
class OpenArmFollowerConfig(RobotConfig):
    # Port to connect to the arm
    right_port: str
    left_port:  str
    
    enable_fd: bool
    
    model_path: str
    
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    motor_types: list[MotorType] = field(default_factory=lambda: [
        MotorType.DM8009, MotorType.DM8009,
        MotorType.DM4340, MotorType.DM4340,
        MotorType.DM4310, MotorType.DM4310, MotorType.DM4310
    ])
    
    send_ids = [ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 ]
    recv_ids = [ 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17 ]
    
    gripper_motor_type = MotorType.DM4310
    gripper_motor_send_id = 0x08
    gripper_motor_recv_id = 0x18
