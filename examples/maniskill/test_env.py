import argparse
import gymnasium as gym
from PourManiskill.src.pour_mani.env.scene.pourwater import PourWaterEnv
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
import cv2
from typing import List
import torch
import os


def save_video(images: List[torch.Tensor], output_path: str, fps: int = 30):
    """Save a list of images as a video file.

    Args:
        images: List of torch tensors or numpy arrays representing RGB images
        output_path: Path where the video will be saved
        fps: Frames per second for the output video
    """
    if not images:
        print("No images to save")
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get the first frame and convert to numpy if needed
    first_frame = images[0]
    if isinstance(first_frame, torch.Tensor):
        first_frame = first_frame.cpu().numpy()

    # Remove batch dimension and get image dimensions
    if first_frame.shape[0] == 1:  # If there's a batch dimension
        first_frame = first_frame.squeeze(0)
    height, width = first_frame.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for i, image in enumerate(images):
            # Print shapes for debugging (only first frame)
            if i == 0:
                print(f"Original image shape: {image.shape}")

            # Convert torch tensor to numpy if needed
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
                if i == 0:
                    print(f"After numpy conversion: {image.shape}")

            # Remove batch dimension if present
            if image.shape[0] == 1:
                image = image.squeeze(0)
                if i == 0:
                    print(f"After removing batch dimension: {image.shape}")

            # Ensure the image is in uint8 format with correct range [0, 255]
            if image.dtype != np.uint8:
                if image.max() <= 1.0:  # Assume [0, 1] range
                    image = (image * 255).astype(np.uint8)
                else:  # Assume already in [0, 255] range
                    image = image.astype(np.uint8)

            # OpenCV uses BGR format
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if i == 0:
                print(f"Final BGR image shape: {bgr_image.shape}")
            out.write(bgr_image)
    finally:
        out.release()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a policy on ManiSkill environments")

    # Policy configuration
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi0_fast_maniskill_state_ee_delta_6d_flow_low_mem_finetune",
        choices=[
            "pi0_fast_maniskill_state_ee_delta_6d_flow_low_mem_finetune",
        ],
        help="Name of the policy configuration",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/root/openpi/checkpoints/pi0_fast_maniskill_state_ee_delta_6d_flow_low_mem_finetune/pi0_fast_state/29999",
        help="Path to the policy checkpoint",
    )

    parser.add_argument(
        "--flow-mode",
        action="store_false",
        help="Use flow mode for the policy",
    )

    # Environment configuration
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
        default="pour water to the cup",
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
        default="/root/openpi/examples/maniskill/pour_water/pour.h5",
        help="Path to the h5 file",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="/root/openpi/examples/maniskill/pour_water/videos/test.mp4",
        help="Directory to save videos",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for output video",
    )
    parser.add_argument("--traj-num", type=int, default=0, help="Trajectory number to evaluate")
    return parser.parse_args()


def main(args):
    env = gym.make(
        id="PourWater-v0",
        obs_mode="rgb",
        control_mode=args.control_mode,
        render_mode="human" if args.render else None,
        max_episode_steps=args.max_steps_per_episode,
    )
    
    obs, _ = env.reset()
    frames = [env.unwrapped.render_rgb_array()]
    for _ in range(100):
        sample = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(sample)
        frames.append(env.unwrapped.render_rgb_array())
        if terminated:
            break
    
    save_video(frames, args.video_dir, args.fps)


if __name__ == "__main__":
    args = parse_args()
    main(args)