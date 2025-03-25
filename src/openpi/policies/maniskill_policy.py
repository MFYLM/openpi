import dataclasses
from typing import Literal

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.utils import _parse_image

from scipy.spatial.transform import Rotation
# ============================================================================
# Example Generation
# ============================================================================


def make_maniskill_example() -> dict:
    """Creates a random input example for the Maniskill policy.

    Returns:
        A dictionary with the expected input structure for ManiskillPolicy,
        including observation/state, observation/image, and prompt.
    """
    # Create random base image (128x128x3 as shown in the dataset conversion script)
    base_image = np.random.randint(256, size=(128, 128, 3), dtype=np.uint8)

    return {
        "observation/state": np.random.rand(9),  # 9-dimensional state vector
        "observation/image": base_image,  # Single RGB image
        "actions": np.random.uniform(-1, 1, size=(7,)),  # 7-dimensional action vector
        "prompt": "pick the red cube and move to the goal target shown in green sphere",
    }


# ============================================================================
# Transform Classes
# ============================================================================


@dataclasses.dataclass(frozen=True)
class ManiskillInputs(transforms.DataTransformFn):
    """Transform Maniskill data format to model input format.

    Args:
        action_dim: The action dimension of the model
        model_type: Determines which model will be used
        control_mode: Control mode - "joint_pos" or "ee_delta_pose"
    """

    # The action dimension of the model
    action_dim: int

    # Determines which model will be used
    model_type: _model.ModelType = _model.ModelType.PI0

    # Control mode: "joint_pos" or "ee_delta_pose" or "ee_delta_pose_6d_flow"
    control_mode: Literal["joint_pos", "ee_delta_pose", "ee_delta_pose_6d_flow"] = "ee_delta_pose"

    def __call__(self, data: dict) -> dict:
        # Get state and pad to model action dimension
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Get base image
        base_image = _parse_image(data["observation/image"])

        # Determine whether to mask padding images
        mask_padding = self.model_type == _model.ModelType.PI0

        # Create input dictionary with required structure
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Add actions if available (for training)
        if "actions" in data:
            actions = data["actions"]
            # Pad actions to model dimension
            actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = actions

        # Add prompt if available
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ManiskillOutputs(transforms.DataTransformFn):
    """Transform model output format to Maniskill action format.

    Args:
        control_mode: Control mode - "joint_pos" or "ee_delta_pose"
    """

    # Control mode: "joint_pos" or "ee_delta_pose" or "ee_delta_pose_6d_flow"
    control_mode: Literal["joint_pos", "ee_delta_pose", "ee_delta_pose_6d_flow"] = "ee_delta_pose"

    def __call__(self, data: dict) -> dict:
        # Extract appropriate action dimensions based on control mode
        if self.control_mode == "joint_pos":
            actions = np.asarray(data["actions"][:, :8])
        elif self.control_mode == "ee_delta_pose":
            actions = np.asarray(data["actions"][:, :7])
        elif self.control_mode == "ee_delta_pose_6d_flow":
            actions = np.asarray(data["actions"][:, :6])
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        return {"actions": actions}


def convert_sapien_pose_to_6d(pose_with_quat: np.ndarray) -> np.ndarray:
    """
    Convert poses with quaternion representation to xyz position and euler angles.

    Args:
        pose_with_quat: numpy array of shape (n, 7) where each row is [x, y, z, qw, qx, qy, qz]
                        (position and quaternion in wxyz order)

    Returns:
        numpy array of shape (n, 6) where each row is [x, y, z, euler_x, euler_y, euler_z]
    """
    n = pose_with_quat.shape[0]
    pose_xyz_euler = np.zeros((n, 6))

    # Copy xyz position
    pose_xyz_euler[:, :3] = pose_with_quat[:, :3]

    # Convert quaternion (wxyz) to Euler angles (xyz)
    # Note: scipy expects xyzw quaternion order, but input has wxyz order
    quats = np.roll(pose_with_quat[:, 3:7], -1, axis=1)  # Convert wxyz to xyzw
    rotations = Rotation.from_quat(quats)
    euler_angles = rotations.as_euler("xyz", degrees=False)  # Get euler angles in radians

    pose_xyz_euler[:, 3:] = euler_angles

    return pose_xyz_euler


def convert_xyz_euler_to_sapien_pose(pose_xyz_euler: np.ndarray) -> np.ndarray:
    """
    Convert poses with xyz position and Euler angles to Sapien pose format.

    Args:
        pose_xyz_euler: numpy array of shape (n, 6) where each row is [x, y, z, euler_x, euler_y, euler_z]

    Returns:
        numpy array of shape (n, 7) where each row is [x, y, z, qw, qx, qy, qz]
        (position and quaternion in wxyz order)
    """
    n = pose_xyz_euler.shape[0]
    pose_with_quat = np.zeros((n, 7))

    # Copy xyz position
    pose_with_quat[:, :3] = pose_xyz_euler[:, :3]

    # Convert Euler angles (xyz) to quaternion (xyzw)
    euler_angles = pose_xyz_euler[:, 3:]
    rotations = Rotation.from_euler("xyz", euler_angles, degrees=False)
    quats_xyzw = rotations.as_quat()  # Get quaternions in xyzw order

    # Convert quaternion from xyzw to wxyz order for Sapien
    quats_wxyz = np.roll(quats_xyzw, 1, axis=1)  # Convert xyzw to wxyz

    # Set quaternion components
    pose_with_quat[:, 3:] = quats_wxyz

    return pose_with_quat


import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    # Create a valid rotation using scipy
    random_rotation = Rotation.random(1)
    quat_xyzw = random_rotation.as_quat()
    quat_wxyz = np.roll(quat_xyzw, 1, axis=1)  # Convert to wxyz

    # Create a valid pose with the proper quaternion
    pose_with_quat = np.zeros((1, 7))
    pose_with_quat[0, :3] = np.random.rand(3)  # Random position
    pose_with_quat[0, 3:] = quat_wxyz[0]  # Set quaternion part

    print(f"Original pose_with_quat: {pose_with_quat}")

    # Convert to xyz_euler and back
    pose_xyz_euler = convert_sapien_pose_to_6d(pose_with_quat)
    print(f"pose_xyz_euler: {pose_xyz_euler}")

    converted_pose = convert_xyz_euler_to_sapien_pose(pose_xyz_euler)
    print(f"Converted pose_with_quat: {converted_pose}")

    # Compare original and converted rotation matrices
    original_rot = Rotation.from_quat(np.roll(pose_with_quat[0, 3:], -1))
    converted_rot = Rotation.from_quat(np.roll(converted_pose[0, 3:], -1))

    print(f"Rotation difference: {np.abs(original_rot.as_matrix() - converted_rot.as_matrix()).max()}")
