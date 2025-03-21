import numpy as np

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


# Added for testing 6d pose prediction
def test_fast_state_tokenizer():
    prompt = "Hello, world!"
    state = {
        "robot_state": np.random.rand(1, 5).astype(np.float32),
        "obj_pose": np.random.rand(1, 6).astype(np.float32),
        "ee_pose": np.random.rand(1, 6).astype(np.float32),
    }
    next_obj_pose = np.random.rand(1, 6).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize_with_additional_state(prompt, state, next_obj_pose)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    pred_obj_pose = tokenizer.extract_obj_pose(tokens)
    assert pred_obj_pose.shape[-1] == 6
    
    
if __name__ == "__main__":
    test_fast_tokenizer()
    test_fast_state_tokenizer()
    