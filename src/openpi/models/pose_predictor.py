from attr import dataclass
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from openpi.shared import array_typing as at
from jaxtyping import Array
from openpi.models import model as _model
from typing_extensions import override

import dataclasses


@dataclasses.dataclass
class PredictorConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    in_dim: int
    out_dim: int
    hid_dim: int
    dropout: float
    depth: int
    
    
class PosePredictor(_model.BaseModel):
    def __init__(self, config: PredictorConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        
        self.in_proj = nnx.Linear(config.in_dim, config.hid_dim, rngs=rngs)
        self.layers = nnx.Sequential(
            [
                nnx.Linear(config.hid_dim * 2, config.hid_dim * 2, rngs=rngs) for _ in range(config.depth)
            ]
        )
        self.dropout = nnx.Dropout(config.dropout, rngs=rngs)
        self.out_proj = nnx.Linear(config.hid_dim, config.out_dim, rngs=rngs)
    
    @override
    def compute_loss(
        self, 
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False
    ) -> at.Float[at.Array, "*b ah ad"]:
        pass
    
    @override
    def __call__(self, obs) -> Array[...]:
        """

        Args:
            obs (_type_): _description_

        Returns:
            Array[...]: _description_
        """
        pass