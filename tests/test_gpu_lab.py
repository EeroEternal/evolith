from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)
pytest.importorskip("triton")

from evolith.lab.contracts import KernelConfig, Shape
from evolith.lab.gpu import (
    baseline_call,
    check_output,
    compiled_baseline,
    make_inputs,
    reference,
    verify,
)
from evolith.lab.kernels import launch


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("rows_per_program", [1, 4])
def test_gpu_independent_oracle_strides_poison_and_immutability(residual, rows_per_program):
    assert verify(
        KernelConfig(rows_per_program=rows_per_program),
        Shape(rows=4, hidden=5120, residual=residual), 1e-6, 1907,
    ) >= 24


def test_nan_output_is_rejected():
    with pytest.raises(AssertionError, match="NaN/Inf"):
        check_output(
            torch.full((2, 3), float("nan"), device="cuda", dtype=torch.bfloat16),
            torch.zeros((2, 3), device="cuda", dtype=torch.bfloat16),
        )


@pytest.mark.parametrize("residual", [False, True])
def test_actual_compiled_baseline_matches_independent_oracle(residual):
    x, r, w, _, _ = make_inputs(Shape(rows=4, hidden=5120, residual=residual), 44)
    fn = compiled_baseline(residual)
    check_output(baseline_call(fn, x, r, w, 1e-6), reference(x, r, w, 1e-6))


def test_unsupported_rank_refuses_before_launch():
    with pytest.raises(ValueError, match="2D tensor"):
        launch(
            torch.tensor(1, device="cuda", dtype=torch.bfloat16),
            None, torch.ones(1, device="cuda", dtype=torch.bfloat16), 1e-6, KernelConfig(),
        )
