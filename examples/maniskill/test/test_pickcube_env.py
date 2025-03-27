import gymnasium as gym
import mani_skill.envs
import numpy as np
from scipy.spatial.transform import Rotation


def transform_obs_to_policy(obs: dict, control_mode: str = "pd_ee_delta_pos") -> dict:
    """
    Transform ManiSkill observation to LeRobot format.

    Args:
        obs: Observation from ManiSkill environment
        control_mode: Control mode being used (pd_ee_delta_pos or pd_joint_pos)

    Returns:
        dict: Transformed observation in LeRobot format
    """
    # Extract relevant data from observation
    # NOTE: The openpi does not support the parallel envs, and don't have the batch size dimension in the input
    # So we need to manually add the batch size dimension to the input
    # And it use the numpy not tensor
    rgb_image = obs["sensor_data"]["base_camera"]["rgb"].cpu().numpy().squeeze()  # Shape should be [128, 128, 3]
    qpos = obs["agent"]["qpos"].cpu().numpy().squeeze()  # Shape should be [9]
    tcp_pose = obs["extra"]["tcp_pose"].cpu().numpy().squeeze()  # Shape should be [7] (xyz, quaternion wxyz)

    # Process TCP pose to get euler angles (similar to the H5 conversion)
    tcp_pose_6d = np.zeros(6)
    tcp_pose_6d[:3] = tcp_pose[:3]  # xyz position

    # Convert quaternion (wxyz) to Euler angles (xyz)
    quat = np.roll(tcp_pose[3:7], -1)  # Convert wxyz to xyzw
    rotation = Rotation.from_quat(quat)
    euler_angles = rotation.as_euler("xyz", degrees=False)  # Get euler angles in radians
    tcp_pose_6d[3:] = euler_angles  # euler_x, euler_y, euler_z

    gripper_pos = qpos[-2:]  # gripper position (last 2 elements)

    # Prepare state based on control mode
    if control_mode == "pd_joint_pos":
        state = qpos  # shape: [9] - joint pos (7) + gripper pos (2)
    elif control_mode in ["pd_ee_delta_pose", "pd_ee_delta_pos"]:
        state = np.concatenate([tcp_pose_6d, gripper_pos])  # shape: [8] - eef pose (6) + gripper pos (2)
    import ipdb

    ipdb.set_trace()
    # Return LeRobot format observations
    maniskill_obs = {
        "observation/image": rgb_image,  # RGB image from camera
        "observation/state": state,  # State vector (depends on control mode)
    }

    return maniskill_obs


# Example usage
if __name__ == "__main__":
    # Create the environment
    control_mode = "pd_ee_delta_pose"
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgb",
        control_mode=control_mode,
        render_mode="human",
    )

    # Reset environment
    obs, _ = env.reset()

    # Main loop
    step = 0
    while True:
        # Sample action
        action = env.action_space.sample()

        # Run environment step
        obs, reward, terminated, truncated, info = env.step(action)

        # Transform current observation to LeRobot format
        maniskill_obs = transform_obs_to_policy(obs, control_mode)

        # Print info about the transformed observation
        print(f"\nStep {step}")
        print(f"RGB Image shape: {maniskill_obs['observation/image'].shape}")
        print(f"State shape: {maniskill_obs['observation/state'].shape}")
        print(f"State values: {maniskill_obs['observation/state']}")

        # This transformed observation + action would be used in LeRobot format:
        # NOTE: The input dict is defined by ourselfs in config.py (also refer to make_maniskill_example in maniskill_policy.py)
        maniskill_pi0_input = {
            "image": maniskill_obs["observation/image"],
            "state": maniskill_obs["observation/state"],
            "actions": action,
            "prompt": "pick the red cube and move to the goal target shown in green sphere",
        }

        # Print action info
        print(f"Action shape: {action.shape}")
        print(f"Action values: {action}")

        # Render
        # env.render()
        step += 1

        import ipdb

        ipdb.set_trace()
