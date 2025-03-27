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
REPO_NAME = "physical-intelligence/maniskill-lerobot-state"


def get_dataset_config():
    """
    Returns the dataset configuration for pd_ee_delta_pose control mode.
    """
    return {
        "action_shape": (6,),  # source object delta pose (6) - xyz, euler_x, euler_y, euler_z
        "state_shapes": {
            "robot_state": (18,),  # robot state (9 * 2) - q pos, q vel
            "obj_pose": (12,),  # object state (6) - xyz, euler_x, euler_y, euler_z + target object state (6)
            "ee_pose": (6,),   # end-effector state (6) - xyz, euler_x, euler_y, euler_z
        },
        "state_names": ["state"],
    }

def main(data_path: str, *, control_mode: str = "pd_ee_delta_pose", push_to_hub: bool = False):
    config = get_dataset_config()
    
    # Create dataset with dictionary-style state components
    dataset = LeRobotDataset.create(
        repo_id=f"{REPO_NAME}-{control_mode}",
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "robot_state": {
                "dtype": "float32",
                "shape": config["state_shapes"]["robot_state"],
                "names": ["joint_positions", "joint_velocities"],
            },
            "obj_pose": {
                "dtype": "float32",
                "shape": config["state_shapes"]["obj_pose"],
                "names": ["obj_x", "obj_y", "obj_z", "obj_rot_x", "obj_rot_y", "obj_rot_z"],
            },
            "ee_pose": {
                "dtype": "float32",
                "shape": config["state_shapes"]["ee_pose"],
                "names": ["ee_x", "ee_y", "ee_z", "ee_rot_x", "ee_rot_y", "ee_rot_z"],
            },
            "actions": {
                "dtype": "float32",
                "shape": config["action_shape"],
                "names": ["delta_x", "delta_y", "delta_z", "delta_rot_x", "delta_rot_y", "delta_rot_z"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    with h5py.File(data_path, "r") as h5_file:
        # Iterate through all trajectories in the file
        for traj_key in h5_file.keys():
            if not traj_key.startswith("traj_"):
                continue

            traj = h5_file[traj_key]

            # Get observations, actions, and other data
            obs = traj["obs"]
            env_states = traj["env_states"]
            agent = obs["agent"]
            actions = traj["actions"][()]

            # Extract relevant data
            rgb_images = obs["sensor_data"]["base_camera"]["rgb"][()]  # shape: [n, 128, 128, 3]
            source_obj_pose = env_states["actors"]["coke_can"][:, :7]  # shape: [n, 7]  (position + velocity)
            tcp_pose = obs["extra"]["tcp_pose"][()]
            target_obj_pose = env_states["actors"]["target_goal_site"][:, :7]  # shape: [n, 7]  (position + velocity)
            qpos, qvel = agent["qpos"], agent["qvel"]

            # Convert tcp_pose from [x,y,z,qw,qx,qy,qz] to [x,y,z,euler_x,euler_y,euler_z]
            source_pose_6d = convert_sapien_pose_to_6d(source_obj_pose)
            target_pose_6d = convert_sapien_pose_to_6d(target_obj_pose)
            ee_pose_6d = convert_sapien_pose_to_6d(tcp_pose)

            # Prepare state
            robot_state = np.concatenate([qpos, qvel], axis=1)  # shape: [n, 12]
            obj_pose = np.concatenate([source_pose_6d, target_pose_6d], axis=1)  # shape: [n, 18]

            assert (
                robot_state.shape[1] == config["state_shapes"]["robot_state"][0]
            ), f"Wrong robot state shape: {robot_state.shape}, expected: {config['state_shape']['robot_state']}"
            assert (
                obj_pose.shape[1] == config["state_shapes"]["obj_pose"][0]
            ), f"Wrong obj state shape: {obj_pose.shape}, expected: {config['state_shape']['obj_pose']}"
            assert(
                ee_pose_6d.shape[1] == config["state_shapes"]["ee_pose"][0]
            ), f"Wrong ee state shape: {ee_pose_6d.shape}, expected: {config['state_shape']['ee_pose']}"

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
                        "robot_state": robot_state[t],
                        "obj_pose": obj_pose[t],
                        "ee_pose": ee_pose_6d[t],
                        "actions": source_pose_6d[t + 1] - source_pose_6d[t],
                    }
                )
            # Add the last observation (which has no action)
            dataset.add_frame(
                {
                    "image": rgb_images[-1],
                    "robot_state": robot_state[-1],
                    "obj_pose": obj_pose[-1],
                    "ee_pose": ee_pose_6d[-1],
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
