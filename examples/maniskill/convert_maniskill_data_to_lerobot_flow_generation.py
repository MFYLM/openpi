"""
Script for converting the Maniskill dataset to LeRobot format.

The Maniskill dataset is stored in h5 format with trajectories containing
observations, actions, rewards, and environment states.

Usage:
uv run examples/maniskill/convert_maniskill_data_to_lerobot.py --data_path /path/to/your/maniskill.h5

This script uses end-effector delta pose control (pd_ee_delta_pose) with action shape: 7.

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/maniskill/convert_maniskill_data_to_lerobot.py --data_path /path/to/your/maniskill.h5 --push_to_hub

Note: to run the script, you need to install h5py:
`uv pip install h5py`

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""

import shutil
import h5py
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
from openpi.policies.maniskill_policy import convert_sapien_pose_to_6d

# Name of the output dataset, also used for the Hugging Face Hub
REPO_NAME = "physical-intelligence/maniskill-lerobot"


def get_dataset_config():
    """
    Returns the dataset configuration for pd_ee_delta_pose control mode.
    """
    return {
        "action_shape": (6,),  # source object delta pose (6) - xyz, euler_x, euler_y, euler_z
        "state_shape": (12,),  # source object pose (6) + target object pose (6) - xyz, euler_x, euler_y, euler_z
        "state_names": ["state"],
    }


def main(data_path: str, *, control_mode: str = "pd_ee_delta_pose_6d_flow", push_to_hub: bool = False):
    # Get dataset configuration
    config = get_dataset_config()

    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / f"{REPO_NAME}-{control_mode}"
    if output_path.exists():
        shutil.rmtree(output_path)

    print(f"Dataset will be saved to: {output_path}")
    print(f"Using control mode: {control_mode}")

    # Create LeRobot dataset, define features to store
    dataset = LeRobotDataset.create(
        repo_id=f"{REPO_NAME}-{control_mode}",
        robot_type="panda",  # Maniskill uses Panda robot
        fps=10,  # Adjust as needed for Maniskill
        features={
            "image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": config["state_shape"],
                "names": config["state_names"],
            },
            "actions": {
                "dtype": "float32",
                "shape": config["action_shape"],
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Load the Maniskill h5 file
    with h5py.File(data_path, "r") as h5_file:
        # Iterate through all trajectories in the file
        for traj_key in h5_file.keys():
            if not traj_key.startswith("traj_"):
                continue

            traj = h5_file[traj_key]

            # Get observations, actions, and other data
            obs = traj["obs"]
            env_states = traj["env_states"]
            actions = traj["actions"][()]

            # Extract relevant data
            rgb_images = obs["sensor_data"]["base_camera"]["rgb"][()]  # shape: [n, 128, 128, 3]
            source_pose = env_states["actors"]["cube"][:, :7]  # shape: [n, 7]  (position + velocity)
            target_pose = env_states["actors"]["goal_site"][:, :7]  # shape: [n, 7]  (position + velocity)

            # Convert tcp_pose from [x,y,z,qw,qx,qy,qz] to [x,y,z,euler_x,euler_y,euler_z]
            source_pose_6d = convert_sapien_pose_to_6d(source_pose)
            target_pose_6d = convert_sapien_pose_to_6d(target_pose)

            # Prepare state
            state = np.concatenate([source_pose_6d, target_pose_6d], axis=1)  # shape: [n, 12]

            assert (
                state.shape[1] == config["state_shape"][0]
            ), f"Wrong state shape: {state.shape}, expected: {config['state_shape']}"

            # Determine number of timesteps (T)
            num_timesteps = rgb_images.shape[0]

            # For each timestep, add a frame to the dataset
            for t in range(num_timesteps - 1):  # -1 because actions have T-1 length
                # filter the grasp phase using gripper pos:
                if actions[t][-1] > 0.0:  # 1: open, -1: close
                    continue
                # only record the move phase:
                dataset.add_frame(
                    {
                        "image": rgb_images[t],
                        "state": state[t],
                        "actions": source _pose_6d[t + 1] - source_pose_6d[t],
                    }
                )
            # Add the last observation (which has no action)
            dataset.add_frame(
                {
                    "image": rgb_images[-1],
                    "state": state[-1],
                    # For the last frame, use a zero action (or repeat the last action)
                    "actions": np.zeros_like(source_pose_6d[0]),
                }
            )

            # Save this trajectory as an episode
            task_name = "pick the red cube and move to the goal target shown in green sphere"
            dataset.save_episode(task=task_name)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["maniskill", "panda", "h5", control_mode],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
