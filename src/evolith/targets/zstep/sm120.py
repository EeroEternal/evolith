from __future__ import annotations

from evolith.targets.base import TargetRegistration
from evolith.targets.registry import register_target

SM120_DSV41_TARGET = register_target(
    TargetRegistration(
        engine="zstep",
        hardware="nvidia_sm120",
        model_family="deepseek_v41",
        supported_operator_families=(
            "moe_experts",
            "sparse_attn",
            "mhc_mixes",
            "attn_qkv_o_proj",
            "engram_compressor_indexer",
            "lm_head",
        ),
        evolution_mode="code_rsi",
        entrypoint="evolith.zstep.loop",
    )
)

SM120_QWEN35_TARGET = register_target(
    TargetRegistration(
        engine="zstep",
        hardware="nvidia_sm120",
        model_family="qwen35_moe",
        supported_operator_families=(
            "moe_experts",
            "gdn_recurrent",
            "attn_qkv_o_proj",
            "allreduce",
            "lm_head",
        ),
        evolution_mode="code_rsi",
        entrypoint="evolith.zstep.loop",
    )
)

__all__ = [
    "SM120_DSV41_TARGET",
    "SM120_QWEN35_TARGET",
]
