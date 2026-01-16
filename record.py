from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from .teleoperators.openarm_leader import OpenArmConfig, OpenArmLeader
from .robots.openarm_follower import OpenArmFollowerConfig, OpenArmFollower

camera_config = {
    "front": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30)
}

robot_config = OpenArmFollowerConfig(
    right_port = 'can2',
    left_port  = 'can3',
    
    enable_fd = True,
    
    cameras=camera_config # type: ignore
)

teleop_config = OpenArmConfig(
    right_port = 'can0',
    left_port  = 'can1',
    
    enable_fd = True,
)

robot = OpenArmFollower(robot_config)
teleop_device = OpenArmLeader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)