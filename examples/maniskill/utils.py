def prepare_policy_input(
    maniskill_obs, action=None, prompt="pick the red cube and move to the goal target shown in green sphere"
):
    """
    Prepare input for the policy in the format expected by OpenPI.

    Args:
        maniskill_obs: Transformed observation from ManiSkill
        action: Previous action (optional)
        prompt: Text prompt describing the task

    Returns:
        dict: Input data formatted for the policy
    """
    policy_input = maniskill_obs
    policy_input["prompt"] = prompt

    # Add actions if provided
    if action is not None:
        policy_input["actions"] = action

    return policy_input
