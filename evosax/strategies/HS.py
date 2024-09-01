from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
from ..strategy import Strategy
from ..utils.eigen_decomp import full_eigen_decomp

@struct.dataclass
class EvoState:
    mean: chex.Array
    archive: chex.Array
    fitness_archive: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0

@struct.dataclass
class EvoParams:
    HMS : int = 100 #和声记忆库的大小，即算法在搜索过程中所考虑的解的个数
    HMCR : float = 0.9 #和声记忆考虑率，表示算法在选择过程中考虑和声记忆库中解的概率。
    PAR : float = 0.9 #音高调整率，用于控制算法在搜索过程中进行局部搜索的强度。
    BW : float =  0.02*(VarMax-VarMin)  #% Fret Width (Bandwidth)
    mutate_best_vector: bool = True  # False - 'random'
    num_diff_vectors: int = 1  # [1, 2]
    cross_over_rate: float = 0.9  # cross-over probability [0, 1]
    diff_w: float = 0.8  # differential weight (F) [0, 2]
    init_min: float = -0.1
    init_max: float = 0.1
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max

class DE(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Harmony search  
        Reference:"""
        assert popsize > 6, "HS requires popsize > 6."
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            n_devices=n_devices,
            **fitness_kwargs
        )
        self.strategy_name = "HS"

    @property
    def params_strategy(self) -> EvoParams:
        return EvoParams()

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """
        `initialize` the Harmony search strategy.
        Initialize all population members by randomly sampling
        positions in search-space (defined in `params`).
        """
        initialization = jax.random.uniform(
            rng,
            (self.popsize, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            fitness_archive=jnp.zeros(self.popsize) + 20e10,
            best_member=initialization.mean(axis=0),
        )
        return state    
