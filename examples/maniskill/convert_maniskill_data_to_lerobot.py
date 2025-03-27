"""
Script for converting the Maniskill dataset to LeRobot format.

The Maniskill dataset is stored in h5 format with trajectories containing
observations, actions, rewards, and environment states.

Usage:
uv run examples/maniskill/convert_maniskill_data_to_lerobot.py --data_path /path/to/your/maniskill.h5 --control_mode pd_joint_pos

Supported control modes:
- pd_joint_pos: Joint position control (action shape: 7)
- pd_ee_delta_pose: End-effector delta pose control (action shape: 8)

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/maniskill/convert_maniskill_data_to_lerobot.py --data_path /path/to/your/maniskill.h5 --control_mode pd_joint_pos --push_to_hub

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

# Name of the output dataset, also used for the Hugging Face Hub
REPO_NAME = "physical-intelligence/maniskill-lerobot"


def get_dataset_config(control_mode):
    """
    Returns the dataset configuration based on the control mode.
    """
    if control_mode == "pd_joint_pos":
        return {
            "action_shape": (8,),  # joint pos (7) + gripper pos (1)
            "state_shape": (9,),  # joint pos (7) + gripper pos (2)
            "state_names": ["state"],
        }
    elif control_mode == "pd_ee_delta_pose":
        return {
            "action_shape": (7,),  # eef delta pose (6) + gripper pos (1)
            "state_shape": (8,),  # eef pose (x,y,z,euler_x,euler_y,euler_z) + gripper pos (2)
            "state_names": ["state"],
        }
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}")


def main(data_path: str, *, control_mode: str = "pd_joint_pos", push_to_hub: bool = False):
    # Validate control mode
    if control_mode not in ["pd_joint_pos", "pd_ee_delta_pose"]:
        raise ValueError(f"Unsupported control mode: {control_mode}. Choose from 'pd_joint_pos' or 'pd_ee_delta_pose'")

    # Get dataset configuration based on control mode
    config = get_dataset_config(control_mode)

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
            actions = traj["actions"][()]  # shape: [n-1, 7] or [n-1, 8]

            # Extract relevant data
            rgb_images = obs["sensor_data"]["base_camera"]["rgb"][()]  # shape: [n, 128, 128, 3]
            qpos = obs["agent"]["qpos"][()]  # shape: [n, 9]
            tcp_pose = obs["extra"]["tcp_pose"][()]  # shape: [n, 7]

            # get the tcp pose in xyz, euler_x,euler_y,euler_z
            tcp_pose_6d = np.zeros((tcp_pose.shape[0], 6))
            tcp_pose_6d[:, :3] = tcp_pose[:, :3]  # xyz position

            # Convert quaternion (wxyz) to Euler angles (xyz)
            # Note: scipy expects xyzw quaternion order, but tcp_pose has wxyz order
            quats = np.roll(tcp_pose[:, 3:7], -1, axis=1)  # Convert wxyz to xyzw
            rotations = Rotation.from_quat(quats)
            euler_angles = rotations.as_euler("xyz", degrees=False)  # Get euler angles in radians

            tcp_pose_6d[:, 3:] = euler_angles  # euler_x, euler_y, euler_z

            gripper_pos = qpos[:, -2:]  # shape: [n, 2]

            # Prepare state based on control mode
            if control_mode == "pd_joint_pos":
                state = qpos
            elif control_mode == "pd_ee_delta_pose":
                # Combine qpos, source_pose, and target_pose
                state = np.concatenate([tcp_pose_6d, gripper_pos], axis=1)  # shape: [n, 7]

            assert (
                state.shape[1] == config["state_shape"][0]
            ), f"Wrong state shape: {state.shape}, expected: {config['state_shape']}"

            # Determine number of timesteps (T)
            num_timesteps = rgb_images.shape[0]

            # For each timestep, add a frame to the dataset
            for t in range(num_timesteps - 1):  # -1 because actions have T-1 length
                import ipdb

                # ipdb.set_trace()
                dataset.add_frame(
                    {
                        "image": rgb_images[t],
                        "state": state[t],
                        "actions": actions[t],
                    }
                )
            # Add the last observation (which has no action)
            dataset.add_frame(
                {
                    "image": rgb_images[-1],
                    "state": state[-1],
                    # For the last frame, use a zero action (or repeat the last action)
                    "actions": actions[-1],
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
