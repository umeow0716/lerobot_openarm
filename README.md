# lerobot_openarm

Please install uv first. See Installation for details.
 [Installation](https://docs.astral.sh/uv/getting-started/installation/).

Clone this repo, and sync the environments automatically.

```
git clone https://github.com/umeow0716/lerobot_openarm.git
uv sync
source .venv/bin/activate
```

Ensure the robotic arm is powered on and correctly plugged in. Then, run this script to enable the CAN interface:

```
cd ~/openarm_can/setup
sudo ./my_arm
```

### Record an eposide 
```
python record.py
```

### Training
```
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=ethanCSL/20260120-must-success \
    --batch_size=16 \
    --steps=20000 \
    --output_dir=outputs/train/svla_multiblock \
    --job_name=my_smolvla_training \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=20260120-must-success \
    --rename_map='{"observation.images.right_camera": "observation.images.camera1", "observation.images.left_camera": "observation.images.camera2", "observation.images.body_camera": "observation.images.camera3"}'

``` 

### Deploy the policy
```
python deploy_ACT.py
python deploy_smolvla.py
```
