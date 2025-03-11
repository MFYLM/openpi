from typing import override
import encoder_base as _model
from attr import dataclass
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from openpi.shared import array_typing as at
from jaxtyping import Array

from im2Flow2Act.tapnet.kubric.test.test_traits import obj

class ObjStateEncoderConfig(_model.BaseEncoderConfig):
    encoder_type: _model.EncoderType = _model.EncoderType.OBJ_POSE
    obj_pose: at.Float[_model.ArrayT, "*b 6"]
    action_dim: int
    hid_dim: int
    out_dim: int
    
    @override
    def encoder_type(self) -> _model.EncoderType:
        return _model.EncoderType.OBJ_POSE
    
    @override
    def create(self, rng: at.KeyArrayLike) -> "ObjStateEncoder":
        return ObjStateEncoder(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.EncoderInput]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            # convert robot state to act_dim. should we do the same to obj_pose?
            observation_spec = _model.EncoderInput(
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                obj_pose=jax.ShapeDtypeStruct([batch_size, 6], jnp.float32),
            )
        return observation_spec
    

class ObjStateEncoder(_model.BaseEncoder):
    
    def __init__(self, config: ObjStateEncoderConfig, rng: at.KeyArrayLike):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        rng, model_rng = jax.random.split(rng)
        self.model = nnx.Sequential(
            nnx.Linear(6, config.hid_dim, key=model_rng),
            nnx.LayerNorm(num_features=config.hid_dim, rngs=model_rng),
            nnx.gelu,
            nnx.Linear(config.hid_dim, config.out_dim, key=model_rng),
        )
        
    def __call__(self, input: _model.EncoderInput) -> at.Float[_model.ArrayT, "*b out_dim"]:
        robot_state, obj_pose = input.state, input.obj_pose
        x = jnp.concatenate([robot_state, obj_pose], axis=-1)
        pose_embed = self.model(x)
        return pose_embed
        
    @at.typecheck
    def compute_loss(
        self, 
        rng: at.KeyArrayLike,
        observation: _model.EncoderInput,
        *,
        train: bool = False
    ):
        raise NotImplementedError


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)    
    config = ObjStateEncoderConfig(
        encoder_type=_model.EncoderType.OBJ_POSE,
        obj_pose=jnp.zeros((1, 6)),
        hid_dim=64,
        out_dim=64
    )
    
    print(f"inspect input: {config.inputs_spec()}")
    fake_obs = config.fake_obs()
    
    encoder = ObjStateEncoder(config, rng=rng)
    print(f"inspect output: {encoder(fake_obs).shape}")