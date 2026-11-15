"""Experimental cache-ownership repair, selected only by the trusted launcher."""

import os
from pathlib import Path

from evolith.lab.rank_cache import isolate_rank_cache
from evolith.lab.store import write_json
from evolith.lab.trace_worker import KernelTraceWorker


class RankLocalWorker(KernelTraceWorker):
    def init_device(self):
        root = os.environ.get("EVOLITH_RANK_CACHE_ROOT")
        directory = os.environ.get("EVOLITH_KERNEL_TRACE_DIR")
        if not root or not directory:
            raise ValueError("Explicit repair namespace and diagnostic output are required.")
        # Must precede normal device setup and subsequent compiled model loading.
        report = isolate_rank_cache(Path(root), self.rank)
        write_json(Path(directory) / f"rank-{self.rank}-cache.json", report)
        return super().init_device()
