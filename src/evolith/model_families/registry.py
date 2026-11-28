from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFamilySpec:
    """Architecture-level operator inventory and serving metadata for a model family."""

    name: str
    architecture: str
    operator_families: tuple[str, ...]
    default_speculative_method: str
    default_num_spec_tokens: int
    page_block_tokens: int
    parity_prompt_ids: tuple[int, ...] = (671, 6102, 294, 8760, 344)


_MODEL_REGISTRY: dict[str, ModelFamilySpec] = {}


def register_model_family(spec: ModelFamilySpec) -> ModelFamilySpec:
    _MODEL_REGISTRY[spec.name] = spec
    return spec


def get_model_family(name: str) -> ModelFamilySpec:
    if name not in _MODEL_REGISTRY:
        available = sorted(_MODEL_REGISTRY)
        raise KeyError(f"Unknown model family {name!r}; available: {available}")
    return _MODEL_REGISTRY[name]


def list_model_families() -> list[ModelFamilySpec]:
    return [_MODEL_REGISTRY[k] for k in sorted(_MODEL_REGISTRY)]


register_model_family(
    ModelFamilySpec(
        name="deepseek_v41",
        architecture="DeepSeek-V4.1-Flash (MLA/SparseAttn + MXFP4 MoE + mHC + Engram)",
        operator_families=(
            "moe_experts",
            "sparse_attn",
            "mhc_mixes",
            "icn_allreduce",
            "attn_qkv_o_proj",
            "engram_compressor_indexer",
            "lm_head",
        ),
        default_speculative_method="dspark",
        default_num_spec_tokens=5,
        page_block_tokens=128,
    )
)

register_model_family(
    ModelFamilySpec(
        name="qwen35_moe",
        architecture="Qwen3.5 MoE (GatedDeltaNet + MoE + MTP)",
        operator_families=(
            "moe_experts",
            "gdn_recurrent",
            "attn_qkv_o_proj",
            "allreduce",
            "lm_head",
        ),
        default_speculative_method="mtp",
        default_num_spec_tokens=3,
        page_block_tokens=16,
    )
)
