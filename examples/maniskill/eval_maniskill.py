import argparse
import time
import gymnasium as gym
import mani_skill.envs
import numpy as np
from scipy.spatial.transform import Rotation
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
# from examples.maniskill.utils import prepare_policy_input


def prepare_policy_input(maniskill_obs, action=None, prompt=None):
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
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}")

    # Return LeRobot format observations
    return {
        "observation/image": rgb_image,  # RGB image from camera
        "observation/state": state,  # State vector (depends on control mode)
    }


def create_policy(config_name, checkpoint_path):
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
    if checkpoint_path.startswith("s3://"):
        checkpoint_dir = download.maybe_download(checkpoint_path)
    else:
        checkpoint_dir = checkpoint_path

    # Create the trained policy
    return policy_config.create_trained_policy(cfg, checkpoint_dir)


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

    # Transform observation to policy format
    maniskill_obs = transform_obs_to_policy(obs, args.control_mode)

    # Main evaluation loop
    step = 0
    env_step = 0
    num_episodes = 0
    episode_rewards = []
    current_episode_reward = 0

    # Initialize viewer if rendering
    viewer = env.render() if args.render else None
    if viewer:
        viewer.paused = args.pause_render

    print(f"Starting Episode {num_episodes+1}")

    while step < args.max_steps:
        # Check if we need to reset the environment
        if env_step >= args.max_steps_per_episode:
            print(f"Completed Episode {num_episodes+1} after {env_step} steps with reward {current_episode_reward}")
            episode_rewards.append(current_episode_reward)
            num_episodes += 1

            if num_episodes >= args.num_episodes:
                break

            print(f"Starting Episode {num_episodes+1}")

            # Reset environment
            obs, _ = env.reset()
            current_episode_reward = 0

            # Transform observation to policy format after reset
            maniskill_obs = transform_obs_to_policy(obs, args.control_mode)

            # Reset environment step counter
            env_step = 0

        # Prepare input for policy
        policy_input = prepare_policy_input(maniskill_obs, prompt=args.task_prompt)

        if args.verbose:
            start_time = time.time()

        # Get action from policy
        actions = run_inference(policy, policy_input)

        if args.verbose:
            end_time = time.time()
            print(f"Time taken to get actions: {end_time - start_time:.4f} seconds")

        # Execute actions
        for action_idx, action in enumerate(actions[: args.max_steps_per_action]):
            if args.control_mode == "pd_joint_pos":
                action[-1] = np.clip(action[-1], -1.0, 1.0)

            # Execute action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            current_episode_reward += reward

            if args.render:
                env.render()

            # Increment environment step counter
            env_step += 1

            if terminated or truncated:
                print(f"Episode {num_episodes+1} terminated early after {env_step} steps")
                print(f"Final reward: {current_episode_reward}")
                episode_rewards.append(current_episode_reward)
                num_episodes += 1

                if num_episodes >= args.num_episodes:
                    break

                # Reset for next episode
                obs, _ = env.reset()
                current_episode_reward = 0
                maniskill_obs = transform_obs_to_policy(obs, args.control_mode)
                env_step = 0
                print(f"Starting Episode {num_episodes+1}")
                break

            if env_step >= args.max_steps_per_episode:
                break

        # Transform new observation if the episode didn't terminate
        if not (terminated or truncated) and env_step < args.max_steps_per_episode:
            maniskill_obs = transform_obs_to_policy(obs, args.control_mode)

        step += 1

    # Print evaluation summary
    print(f"\nEvaluation Summary:")
    print(f"Completed {num_episodes} episodes")
    if episode_rewards:
        print(f"Average reward: {np.mean(episode_rewards):.4f}")
        print(f"Min reward: {np.min(episode_rewards):.4f}")
        print(f"Max reward: {np.max(episode_rewards):.4f}")

    # Close environment
    env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a policy on ManiSkill environments")

    # Policy configuration
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi0_fast_maniskill_ee_delta_pose_low_mem_finetune",
        choices=[
            "pi0_fast_maniskill_ee_delta_pose_low_mem_finetune",
            "pi0_fast_maniskill_joint_pos_low_mem_finetune",
        ],
        help="Name of the policy configuration",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/home/haoyang/project/haoyang/openpi/checkpoints/pi0_fast_maniskill_ee_delta_pose_low_mem_finetune/pickcube_ms_ee_delta_pose/20000",
        help="Path to the policy checkpoint",
    )

    # Environment configuration
    parser.add_argument("--env-id", type=str, default="PickCube-v1", help="ManiSkill environment ID")
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
        "--max-steps-per-action", type=int, default=10, help="Maximum number of steps to execute per policy action"
    )
    parser.add_argument("--num-episodes", type=int, default=20, help="Number of episodes to evaluate")

    # Rendering options
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--pause-render", action="store_true", help="Pause rendering initially")

    # Misc
    parser.add_argument("--verbose", action="store_true", help="Print verbose information")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_policy(args)
