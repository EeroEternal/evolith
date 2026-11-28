from __future__ import annotations

from evolith.lab.zstep_sail import (
    DEFAULT_SAIL_CONTAINER_IMAGE,
    SailPolicy,
    ZstepSailEvolution,
    ZstepSailEvolutionConfig,
)
from evolith.targets.base import TargetRegistration
from evolith.targets.registry import register_target

SAIL_DSV41_TARGET = register_target(
    TargetRegistration(
        engine="zstep",
        hardware="thead_ppu_zw810e",
        model_family="deepseek_v41",
        supported_operator_families=(
            "moe_experts",
            "sparse_attn",
            "mhc_mixes",
            "icn_allreduce",
            "attn_qkv_o_proj",
            "engram_compressor_indexer",
            "lm_head",
        ),
        evolution_mode="policy_search",
        entrypoint="evolith.lab.zstep_sail:ZstepSailEvolution",
    )
)

__all__ = [
    "DEFAULT_SAIL_CONTAINER_IMAGE",
    "SAIL_DSV41_TARGET",
    "SailPolicy",
    "ZstepSailEvolution",
    "ZstepSailEvolutionConfig",
]
