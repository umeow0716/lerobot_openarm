from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from teleoperators.openarm_leader import OpenArmConfig, OpenArmLeader
from robots.openarm_follower import OpenArmFollowerConfig, OpenArmFollower

camera_config = {}

robot_config = OpenArmFollowerConfig(
    right_port = 'can2',
    left_port  = 'can3',
    
    enable_fd = True,
    
    model_path='/home/csl/lerobot_openarm/model/openarm_description.urdf',
    
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

try:
    while True:
        observation = robot.get_observation()
        action = teleop_device.get_action()
        robot.send_action(action)
except KeyboardInterrupt:
    pass

robot.disconnect()
teleop_device.disconnect()