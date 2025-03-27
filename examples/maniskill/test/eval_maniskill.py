import numpy as np
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download


def create_policy(config_name, checkpoint_path):
    """Create a trained policy from a config and checkpoint."""
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
    """Run inference with the given policy on the provided data."""
    # Run inference
    result = policy.infer(example_data)

    # Return the actions
    return result["actions"]


def evaluate_policy(config_name, checkpoint_path, example_data):
    """Legacy function that combines policy creation and inference."""
    policy = create_policy(config_name, checkpoint_path)
    return run_inference(policy, example_data)


from openpi.policies.maniskill_policy import make_maniskill_example

# Example usage
if __name__ == "__main__":
    # Configuration
    config_name = "pi0_fast_maniskill_ee_delta_pose_low_mem_finetune"  # replace with your config name
    checkpoint_path = "/home/haoyang/project/haoyang/openpi/checkpoints/pi0_fast_maniskill_ee_delta_pose_low_mem_finetune/pickcube_ms_ee_delta_pose/20000"  # replace with your checkpoint path

    # Create example data - replace these placeholder values with your actual data
    example_data = make_maniskill_example()
    # Decoupled approach
    policy = create_policy(config_name, checkpoint_path)
    import ipdb

    ipdb.set_trace()
    actions = run_inference(policy, example_data)

    # Print results
    print("Action output shape:", actions.shape)
    print("First few values of action:", actions[:5])
