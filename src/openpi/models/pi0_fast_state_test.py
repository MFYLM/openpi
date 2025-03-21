import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.pi0_fast_state as _pi0

import numpy as np
from openpi.models.pi0_fast_state import Pi0FASTStateConfig, Pi0FASTState
from openpi.models import model as _model
import dataclasses
from openpi.shared import nnx_utils



def _get_frozen_state(config: _pi0.Pi0FASTStateConfig) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_fast_full_finetune():
    config = _pi0.Pi0FASTStateConfig()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_fast_gemma_lora():
    config = _pi0.Pi0FASTStateConfig(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0__fast_all_lora():
    config = _pi0.Pi0FASTStateConfig(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_sample_obj_states():
    key = jax.random.key(0)
    config = _pi0.Pi0FASTStateConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)
    batch_size = 2
    obs = config.fake_obs(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, obs.state["obj_pose"])
    assert loss.shape == (batch_size,)

    obj_pose = nnx_utils.module_jit(model.sample_obj_states)(key, obs)
    print(f"obj_pose: {obj_pose}")
    assert obj_pose.shape == (batch_size, 6)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


if __name__ == "__main__":
    test_pi0_fast_full_finetune()
    test_pi0_fast_gemma_lora()
    test_pi0__fast_all_lora()
    test_sample_obj_states()
