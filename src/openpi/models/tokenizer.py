import logging

from cv2 import log
import numpy as np
import sentencepiece
from transformers import AutoProcessor

import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # tokenize "\n" separately as the "start of answer" token
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens
        self._prev_obj_state = None  # Keep track of previous object state

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)
        
        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|")
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)
    
    # Added for enabling 6d pose prediction
    def tokenize_with_additional_state(
        self, 
        prompt: str, 
        state: dict[str, np.ndarray],
        next_obj_state: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")
        
        robot_state = state["robot_state"]
        obj_state = state["obj_state"]
        ee_pose = state["ee_pose"]
        
        # Discretize input state
        discretized_state = np.digitize(robot_state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        robot_state_str = " ".join(map(str, discretized_state))
        discretized_obj = np.digitize(obj_state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        obj_str = " ".join(map(str, discretized_obj))
        discretized_ee = np.digitize(ee_pose, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        ee_str = " ".join(map(str, discretized_ee))
        
        prefix = f"Task: {cleaned_text}, State: {robot_state_str}, EE Pose: {ee_str}, Object State: {obj_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)
        
        postfix_tokens = []
        if next_obj_state is not None:
            # Tokenize next state
            discretized_next = np.digitize(next_obj_state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            next_state_str = " ".join(map(str, discretized_next))
            next_state_tokens = self._paligemma_tokenizer.encode(next_state_str)
            
            # generate pose flow first
            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                # self._paligemma_tokenizer.encode("Action: ")
                # + action_tokens_in_pg.tolist()
                # + self._paligemma_tokenizer.encode("|")
                self._paligemma_tokenizer.encode("Object State: ")
                + next_state_tokens
                + self._paligemma_tokenizer.encode("|")
            )
            
        
        # Combine tokens and create masks
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)

        # Pad tokens to max length
        pad_len = self._max_len - len(tokens)
        if pad_len > 0:
            padding = [False] * pad_len
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                )
            tokens = tokens[:self._max_len]
            token_mask = token_mask[:self._max_len]
            ar_mask = ar_mask[:self._max_len]
            loss_mask = loss_mask[:self._max_len]

        return (
            np.array(tokens),
            np.array(token_mask),
            np.array(ar_mask),
            np.array(loss_mask),
        )

    def extract_obj_state(self, tokens: np.ndarray) -> np.ndarray:
        # Decode tokens to text
        decoded = self._paligemma_tokenizer.decode(tokens.tolist())
        
        # Extract object state string
        if "Object State: " not in decoded or "|" not in decoded:
            return np.zeros(6, dtype=np.float32)
        
        state_part = decoded.split("Object State: ")[1].split("|")[0].strip()
        try:
            discretized = list(map(int, state_part.split()))
        except ValueError:
            logging.warning("Failed to extract object state from tokens!")
            return np.zeros(6, dtype=np.float32)
        
        # Convert to continuous values
        bins = np.linspace(-1, 1, 257)  # 257 edges for 256 bins
        midpoints = (bins[:-1] + bins[1:]) / 2
        if len(discretized) != 6:
            return np.zeros(6, dtype=np.float32)
        
        # Clip to valid bin indices and get midpoints
        discretized = np.clip(discretized, 0, 255)
        return np.array([midpoints[i] for i in discretized], dtype=np.float32)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        # only keep tokens that are not in the skip tokens
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
