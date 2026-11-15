"""Private worker cache ownership, without choosing any kernel configuration."""

from __future__ import annotations

import os
from pathlib import Path


def isolate_rank_cache(root: Path, rank: int) -> dict:
    if not root.is_absolute() or type(rank) is not int or rank not in range(4):
        raise ValueError("Rank-local caches require an absolute namespace and rank 0–3.")
    settings = {
        "TORCHINDUCTOR_CACHE_DIR": str(root / f"rank-{rank}" / "inductor"),
        "TRITON_CACHE_DIR": str(root / f"rank-{rank}" / "triton"),
    }
    for value in settings.values():
        Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.update(settings)
    return {"rank": rank, "environment": settings,
            "scope": "rank-local cache ownership; unchanged compiler, autotuning and arithmetic"}
