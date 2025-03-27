import argparse
import time
import gymnasium as gym
import mani_skill.envs
import numpy as np
from scipy.spatial.transform import Rotation
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
from mani_skill.utils.structs import Pose
from openpi.policies.maniskill_policy import convert_sapien_pose_to_6d, convert_xyz_euler_to_sapien_pose
import h5py
import openpi.policies.policy as _policy
from pathlib import Path


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


def prepare_policy_input(maniskill_obs: dict, action: np.ndarray = None, prompt: str = None):
    """
    Prepare input for the policy in the format expected by OpenPI.

    Args:
        maniskill_obs: Transformed observation from ManiSkill
        action: Previous action (optional)
        prompt: Text prompt describing the task

    Returns:
        dict: Input data formatted for the policy
    """
    policy_input = maniskill_obs.copy()

    if prompt:
        policy_input["prompt"] = prompt

    if action is not None:
        policy_input["actions"] = action

    return policy_input


def transform_obs_to_policy(obs: dict, control_mode: str) -> dict:
    """
    Transform ManiSkill observation to LeRobot format.

    Args:
        obs: Observation from ManiSkill environment
        control_mode: Control mode being used (pd_ee_delta_pos, pd_ee_delta_pose, or pd_joint_pos)

    Returns:
        dict: Transformed observation in LeRobot format
    """
    # Extract relevant data from observation
    rgb_image = obs["sensor_data"]["base_camera"]["rgb"].cpu().numpy().squeeze()
    qpos = obs["agent"]["qpos"].cpu().numpy().squeeze()
    tcp_pose = obs["extra"]["tcp_pose"].cpu().numpy().squeeze()
    cube_pose = obs["extra"]["cube_pose"].cpu().numpy()
    goal_pose = obs["extra"]["goal_pose"].cpu().numpy()
    cube_pose_6d = convert_sapien_pose_to_6d(cube_pose).squeeze().astype(np.float32)
    goal_pose_6d = convert_sapien_pose_to_6d(goal_pose).squeeze().astype(np.float32)

    # Process TCP pose to get euler angles
    tcp_pose_6d = np.zeros(6)
    tcp_pose_6d[:3] = tcp_pose[:3]  # xyz position

    # Convert quaternion (wxyz) to Euler angles (xyz)
    quat = np.roll(tcp_pose[3:7], -1)  # Convert wxyz to xyzw
    rotation = Rotation.from_quat(quat)
    euler_angles = rotation.as_euler("xyz", degrees=False)
    tcp_pose_6d[3:] = euler_angles

    gripper_pos = qpos[-2:]  # gripper position (last 2 elements)

    # Prepare state based on control mode
    if control_mode == "pd_joint_pos":
        state = qpos  # shape: [9] - joint pos (7) + gripper pos (2)
    elif control_mode in ["pd_ee_delta_pose", "pd_ee_delta_pos"]:
        state = np.concatenate([tcp_pose_6d, gripper_pos])  # shape: [8] - eef pose (6) + gripper pos (2)
    elif control_mode == "pd_ee_delta_pose_6d_flow":
        state = np.concatenate(
            [cube_pose_6d, goal_pose_6d]
        )  # shape: [12] - source object pose (6) + target object pose (6)
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}")

    # Return LeRobot format observations
    return {
        "observation/image": rgb_image,  # RGB image from camera
        "observation/state": state,  # State vector (depends on control mode)
    }


def create_policy(config_name: str, checkpoint_path: str | Path = None) -> _policy.Policy:
    """
    Create a trained policy from a config and checkpoint.

    Args:
        config_name: Name of the configuration to use
        checkpoint_path: Path to the checkpoint (local or S3)

    Returns:
        The trained policy
    """
    # Load config
    cfg = config.get_config(config_name)

    # Download or use local checkpoint
    if isinstance(checkpoint_path, str) and checkpoint_path.startswith("s3://"):
        checkpoint_dir = download.maybe_download(checkpoint_path)
    else:
        checkpoint_dir = Path(checkpoint_path)

    # Create the trained policy
    return policy_config.create_trained_policy(cfg, checkpoint_dir)


def prepare_grasp_phase(
    h5_file_path: str,
    traj_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # load all the demonstrations and select the traj_num-th trajectory
    with h5py.File(h5_file_path, "r") as f:
        traj = f[f"traj_{traj_num}"]
        # get the robot_qpos, source_pose, and target_pose
        robot_qposes = traj["obs"]["agent"]["qpos"][()]
        source_poses = traj["env_states"]["actors"]["cube"][:, :7]
        target_poses = traj["env_states"]["actors"]["goal_site"][:, :7]
        actions = traj["actions"][()]

        grasp_phase_start_idx = np.where(actions[:, -1] == -1)[0][0] - 1
        robot_qpos_before_grasp = robot_qposes[grasp_phase_start_idx]
        source_pose_before_grasp = source_poses[grasp_phase_start_idx]
        target_pose_before_grasp = target_poses[grasp_phase_start_idx]

    return robot_qpos_before_grasp, source_pose_before_grasp, target_pose_before_grasp


def set_grasp_phase(
    env: mani_skill.envs.PickCubeEnv, robot_qpos: np.ndarray, source_pose: np.ndarray, target_pose: np.ndarray
):
    """
    Prepare the grasp phase for the policy.
    """
    # import ipdb
    # ipdb.set_trace()

    env.unwrapped.agent.robot.set_qpos(robot_qpos)
    env.unwrapped.cube.set_pose(Pose.create_from_pq(p=source_pose[:3], q=source_pose[3:]))
    env.unwrapped.goal_site.set_pose(Pose.create_from_pq(p=target_pose[:3], q=target_pose[3:]))
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
    return obs


def run_inference(policy, example_data):
    """
    Run inference with the given policy on the provided data.

    Args:
        policy: The policy to use for inference
        example_data: Input data for the policy

    Returns:
        The actions from the policy
    """
    # Run inference

    result = policy.infer(example_data)
    return result["actions"]


def evaluate_policy(args):
    """
    Evaluate a policy on ManiSkill environments.

    Args:
        args: Command line arguments
    """
    # Create the policy
    policy = create_policy(args.config_name, args.checkpoint_path)
    control_mode = args.control_mode
    if args.flow_mode:
        control_mode = "pd_ee_delta_pose_6d_flow"

    # Create the environment
    env = gym.make(
        args.env_id,
        obs_mode="rgb",
        control_mode=args.control_mode,
        render_mode="human" if args.render else None,
        max_episode_steps=args.max_steps_per_episode,
    )

    # Reset environment
    obs, _ = env.reset()
    if args.flow_mode:
        robot_qpos, source_pose, target_pose = prepare_grasp_phase(args.h5_file_path, args.traj_num)
        obs = set_grasp_phase(env, robot_qpos, source_pose, target_pose)

    # Transform observation to policy format
    maniskill_obs = transform_obs_to_policy(obs, control_mode)

    # Main evaluation loop
    step = 0
    env_step = 0
    num_episodes = 0

    # Initialize viewer if rendering
    viewer = env.render() if args.render else None
    if viewer:
        viewer.paused = args.pause_render

    print(f"Starting Episode {num_episodes+1}")

    while step < args.max_steps:
        # Prepare input for policy
        policy_input = prepare_policy_input(maniskill_obs, prompt=args.task_prompt)

        if args.verbose:
            start_time = time.time()

        # Get action from policy
        actions_6d_delta = run_inference(policy, policy_input)

        if args.verbose:
            end_time = time.time()
            print(f"Time taken to get actions: {end_time - start_time:.4f} seconds")

        # Execute actions
        for action_idx, action_6d_delta in enumerate(actions_6d_delta[: args.max_steps_per_action]):
            # Execute action in environment
            print(f"action: {action_6d_delta}")

            cube_pose = env.unwrapped.cube.pose.raw_pose
            cube_pose_numpy = cube_pose.cpu().numpy()
            cube_pose_6d = convert_sapien_pose_to_6d(cube_pose_numpy).squeeze()
            action_pose_6d = action_6d_delta + cube_pose_6d
            action_pose_6d = action_pose_6d[np.newaxis, :]
            action_pose = convert_xyz_euler_to_sapien_pose(action_pose_6d).squeeze()
            env.unwrapped.cube.set_pose(Pose.create_from_pq(p=action_pose[:3], q=action_pose[3:7]))
            obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))

            if args.render:
                env.render()

            # Increment environment step counter
            env_step += 1

            if env_step >= args.max_steps_per_episode:
                print(f"Episode {num_episodes+1} terminated early after {env_step} steps")
                num_episodes += 1

                if num_episodes >= args.num_episodes:
                    break

                # Reset for next episode
                obs, _ = env.reset()
                # randomize the traj num:
                traj_num = np.random.randint(0, 100)
                if args.flow_mode:
                    robot_qpos, source_pose, target_pose = prepare_grasp_phase(args.h5_file_path, traj_num)
                    obs = set_grasp_phase(env, robot_qpos, source_pose, target_pose)
                maniskill_obs = transform_obs_to_policy(obs, control_mode)
                env_step = 0
                print(f"Starting Episode {num_episodes+1}")
                break

        # Transform new observation if the episode didn't terminate
        if env_step < args.max_steps_per_episode:
            maniskill_obs = transform_obs_to_policy(obs, control_mode)

        step += 1

    # Print evaluation summary
    print(f"\nEvaluation Summary:")
    print(f"Completed {num_episodes} episodes")

    # Close environment
    env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a policy on ManiSkill environments")

    # Policy configuration
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi0_fast_maniskill_ee_delta_6d_flow_low_mem_finetune",
        choices=[
            "pi0_fast_maniskill_ee_delta_6d_flow_low_mem_finetune",
        ],
        help="Name of the policy configuration",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/home/haoyang/project/haoyang/openpi/checkpoints/pi0_fast_maniskill_ee_delta_6d_flow_low_mem_finetune/pickcube_ms_ee_delta_6d_flow/25000",
        help="Path to the policy checkpoint",
    )

    parser.add_argument(
        "--flow-mode",
        action="store_true",
        help="Use flow mode for the policy",
    )

    # Environment configuration
    parser.add_argument("--env-id", type=str, default="PickCube-v1-flow", help="ManiSkill environment ID")
    parser.add_argument(
        "--control-mode",
        type=str,
        default="pd_ee_delta_pose",
        choices=["pd_ee_delta_pose", "pd_ee_delta_pos", "pd_joint_pos"],
        help="Control mode for the environment",
    )
    parser.add_argument(
        "--task-prompt",
        type=str,
        default="pick the red cube and move to the goal target shown in green sphere",
        help="Text prompt describing the task",
    )

    # Evaluation parameters
    parser.add_argument("--max-steps", type=int, default=3000, help="Maximum number of total steps")
    parser.add_argument("--max-steps-per-episode", type=int, default=150, help="Maximum number of steps per episode")
    parser.add_argument(
        "--max-steps-per-action", type=int, default=5, help="Maximum number of steps to execute per policy action"
    )
    parser.add_argument("--num-episodes", type=int, default=20, help="Number of episodes to evaluate")

    # Rendering options
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--pause-render", action="store_true", help="Pause rendering initially")

    # Misc
    parser.add_argument("--verbose", action="store_true", help="Print verbose information")

    parser.add_argument(
        "--h5-file-path",
        type=str,
        default="/home/haoyang/project/haoyang/openpi/examples/maniskill/data/pickcube/pd_ee_delta_pose.h5",
        help="Path to the h5 file",
    )
    parser.add_argument("--traj-num", type=int, default=0, help="Trajectory number to evaluate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_policy(args)
