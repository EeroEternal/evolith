"""Diagnostic-only worker; uses the stock model execution and compilation paths."""

import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker

from evolith.lab.compiler_policy import compiler_policy_report
from evolith.lab.kernel_trace import install
from evolith.lab.store import write_json


class KernelTraceWorker(Worker):
    def init_device(self):
        from torch._inductor import config

        report = compiler_policy_report(
            self.rank, os.environ.get("TORCHINDUCTOR_DETERMINISTIC"), config.deterministic,
        )
        directory = os.environ.get("EVOLITH_KERNEL_TRACE_DIR")
        if not directory:
            raise ValueError("Explicit diagnostic output is required.")
        write_json(Path(directory) / f"rank-{self.rank}-compiler.json", report)
        result = super().init_device()
        mode = os.environ.get("EVOLITH_KERNEL_TRACE_MODE")
        if mode not in ("observe", "off"):
            raise ValueError("Explicit diagnostic trace mode is required.")
        self._evolith_kernel_trace = install(self.rank) if mode == "observe" else None
        return result

    def execute_model(self, scheduler_output):
        if self._evolith_kernel_trace is None:
            return super().execute_model(scheduler_output)
        with self._evolith_kernel_trace.request(scheduler_output):
            return super().execute_model(scheduler_output)
