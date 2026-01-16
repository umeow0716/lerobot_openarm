from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
import os

hf_token = os.environ.get("HF_TOKEN") 

import shutil
shutil.rmtree('/home/csl/.cache/huggingface/lerobot', ignore_errors=True)

if hf_token:
    login(token=hf_token)
    print("Logged in successfully!")
else:
    print("HF_TOKEN environment variable not set. Cannot log in.")

import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from teleoperators.openarm_leader import OpenArmConfig, OpenArmLeader
from robots.openarm_follower import OpenArmFollowerConfig, OpenArmFollower
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

NUM_EPISODES = 3
FPS = 30
EPISODE_TIME_SEC = 10
RESET_TIME_SEC = 2
TASK_DESCRIPTION = "My task description"

# Create robot configuration
camera_config = {
    "right_camera": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS) # Optional: fourcc="MJPG" for troubleshooting OpenCV async error.
}

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

# Initialize the robot and teleoperator
robot = OpenArmFollower(robot_config)
teleop = OpenArmLeader(teleop_config)

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

time.sleep(1.0)  # Allow some time for connections to establish

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action") # type: ignore
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="trash/trash",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
    video_backend='torchcodec'
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Create the required processors
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )
    
    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
dataset.push_to_hub()