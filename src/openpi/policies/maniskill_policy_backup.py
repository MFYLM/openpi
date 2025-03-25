import dataclasses
from typing import Literal

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.utils import _normalize
from openpi.policies.utils import _parse_image
from openpi.policies.utils import _unnormalize

# ============================================================================
# Robot Configuration
# ============================================================================


@dataclasses.dataclass
class RobotConfig:
    """Configuration for robot-specific parameters."""

    # Joint configuration
    joint_flip_mask: np.ndarray
    action_dim: int

    # Gripper configuration
    gripper_angular_min_pi: float
    gripper_angular_max_pi: float
    gripper_angular_min_cust: float
    gripper_angular_max_cust: float
    gripper_linear_min: float
    gripper_linear_max: float
    gripper_arm_length: float
    gripper_horn_radius: float

    @classmethod
    def aloha(cls):
        """Default configuration for aloha robot."""
        return cls(
            joint_flip_mask=np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            action_dim=8,
            gripper_angular_min_pi=0.4,
            gripper_angular_max_pi=1.5,
            gripper_angular_min_cust=-0.6213,
            gripper_angular_max_cust=1.4910,
            gripper_linear_min=0.01844,
            gripper_linear_max=0.05800,
            gripper_arm_length=0.036,
            gripper_horn_radius=0.022,
        )

    @classmethod
    def from_config(cls, config_dict):
        """Create a configuration from a dictionary."""
        return cls(**config_dict)


# Default robot configuration
DEFAULT_ROBOT_CONFIG = RobotConfig.aloha()

# ============================================================================
# Gripper Transformation Functions
# ============================================================================


def _gripper_from_angular(value, config=DEFAULT_ROBOT_CONFIG):
    """Convert from the gripper position used by pi0 to the gripper position used by Aloha."""
    # Unnormalize from [0,1] to robot's angular range
    value = _unnormalize(value, min_val=config.gripper_angular_min_pi, max_val=config.gripper_angular_max_pi)

    # Normalize to Aloha's range
    return _normalize(value, min_val=config.gripper_angular_min_cust, max_val=config.gripper_angular_max_cust)


def _gripper_from_angular_inv(value, config=DEFAULT_ROBOT_CONFIG):
    """Directly inverts the gripper_from_angular function."""
    value = _unnormalize(value, min_val=config.gripper_angular_min_cust, max_val=config.gripper_angular_max_cust)
    return _normalize(value, min_val=config.gripper_angular_min_pi, max_val=config.gripper_angular_max_pi)


def _gripper_to_angular(value, config=DEFAULT_ROBOT_CONFIG):
    """Convert linear gripper positions to angular space for pi0 compatibility."""
    # Unnormalize from [0,1] to robot's linear range
    value = _unnormalize(value, min_val=config.gripper_linear_min, max_val=config.gripper_linear_max)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # Apply the transformation with configurable parameters
    value = linear_to_radian(value, arm_length=config.gripper_arm_length, horn_radius=config.gripper_horn_radius)

    # Normalize to [0, 1] using robot's angular range
    return _normalize(value, min_val=config.gripper_angular_min_pi, max_val=config.gripper_angular_max_pi)


# ============================================================================
# Joint Action Transformation Functions
# ============================================================================


def _joint_flip_mask(config=DEFAULT_ROBOT_CONFIG) -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return config.joint_flip_mask


def _encode_joint_actions(actions: np.ndarray, *, adapt_to_pi: bool = False, config=DEFAULT_ROBOT_CONFIG) -> np.ndarray:
    """Convert joint actions from Maniskill space to Pi space."""
    if adapt_to_pi:
        actions = _joint_flip_mask(config) * actions
        actions[:, -1] = _gripper_from_angular(actions[:, -1], config)

    return actions


def _encode_joint_actions_inv(
    actions: np.ndarray, *, adapt_to_pi: bool = False, config=DEFAULT_ROBOT_CONFIG
) -> np.ndarray:
    """Convert joint actions from Pi space back to Maniskill space."""
    if adapt_to_pi:
        actions = _joint_flip_mask(config) * actions
        actions[:, -1] = _gripper_from_angular_inv(actions[:, -1], config)

    return actions


# ============================================================================
# Data Decoding Functions
# ============================================================================


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False, config=DEFAULT_ROBOT_CONFIG) -> np.ndarray:
    """Transform state representation between Maniskill and Pi spaces."""
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask(config) * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[-1] = _gripper_to_angular(state[-1], config)
    return state


def _decode_maniskill(data: dict, *, adapt_to_pi: bool = False, config=DEFAULT_ROBOT_CONFIG) -> dict:
    """Decode Maniskill data format to a standardized internal format.

    Args:
        data: Dictionary containing Maniskill data
        adapt_to_pi: Whether to adapt values to Pi space

    Returns:
        Dictionary with standardized keys and processed values
    """
    # Process state
    state = np.asarray(data["observation/state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi, config=config)

    # Convert images from Maniskill format to internal format
    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


# ============================================================================
# Example Generation
# ============================================================================


def make_maniskill_example() -> dict:
    """Creates a random input example for the Maniskill policy."""
    return {
        "observation/state": np.random.rand(9),
        "observation/image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
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
        adapt_to_pi: If true, convert values from Maniskill space to Pi space
    """

    # The action dimension of the model
    action_dim: int

    # Determines which model will be used
    model_type: _model.ModelType = _model.ModelType.PI0

    # Control mode: "joint_pos" or "ee_delta_pose"
    control_mode: Literal["joint_pos", "ee_delta_pose"] = "ee_delta_pose"

    # If true, this will convert values from Maniskill space to Pi space
    adapt_to_pi: bool = False

    # Robot configuration
    robot_config: RobotConfig = DEFAULT_ROBOT_CONFIG

    def __call__(self, data: dict) -> dict:
        # Standardize input format
        if self.control_mode == "joint_pos":
            # Convert from Maniskill format to internal format
            data = _decode_maniskill(data, adapt_to_pi=self.adapt_to_pi, config=self.robot_config)

        # Get state and pad to model action dimension
        state = transforms.pad_to_dim(data["state"], self.action_dim)

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

            # Apply transformations for joint position control
            if self.control_mode == "joint_pos" and self.adapt_to_pi:
                actions = _encode_joint_actions_inv(actions, adapt_to_pi=self.adapt_to_pi, config=self.robot_config)

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
        adapt_to_pi: If true, convert values from Pi space to Maniskill space
    """

    # Control mode: "joint_pos" or "ee_delta_pose"
    control_mode: Literal["joint_pos", "ee_delta_pose"] = "ee_delta_pose"

    # If true, this will convert values from Pi space to Maniskill space
    adapt_to_pi: bool = False

    # Robot configuration
    robot_config: RobotConfig = DEFAULT_ROBOT_CONFIG

    def __call__(self, data: dict) -> dict:
        # Extract appropriate action dimensions based on control mode
        if self.control_mode == "joint_pos":
            actions = np.asarray(data["actions"][:, :8])
        elif self.control_mode == "ee_delta_pose":
            actions = np.asarray(data["actions"][:, :7])
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")

        # Apply joint space transformations if using joint position control
        if self.control_mode == "joint_pos" and self.adapt_to_pi:
            actions = _encode_joint_actions(actions, adapt_to_pi=self.adapt_to_pi, config=self.robot_config)

        return {"actions": actions}
