# zstep Source-Level RSI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Evolith CLI workflow that lets Kimi CLI modify isolated zstep source copies and promotes only patches that pass real, independent DeepSeek-V4.1-Flash TP=8 verification.

**Architecture:** A task model and patch policy constrain every candidate. A workspace manager creates Evolith-owned source copies, a Kimi proposer edits one candidate, local gates reject invalid diffs, and an SSH runner executes separate screen/verify directories. SQLite and per-candidate evidence preserve lineage and the accepted patch stack.

**Tech Stack:** Python 3.10+, Pydantic 2, Typer, SQLite, subprocess, existing zstep Rust/Python benchmark tooling, Kimi CLI, SSH/rsync, Docker on the remote host.

**Execution constraint:** Do not run `git commit` or `git push`; the user did not request commits. Do not use sudo. Do not dispatch subagents. Do not modify `~/github/zstep` or remote `~/zstep-s0`.

---

## File map

- Create `src/evolith/zstep/task.py`: task contract, candidate status, benchmark metric models.
- Create `src/evolith/zstep/workspace.py`: isolated clones, accepted patch application, diff allowlist and size checks.
- Create `src/evolith/zstep/archive.py`: SQLite lineage, evidence paths and accepted baseline patch stack.
- Create `src/evolith/zstep/proposer.py`: non-interactive Kimi invocation with scrubbed credentials.
- Create `src/evolith/zstep/gates.py`: local static commands, zstep JSON parsing and acceptance rules.
- Create `src/evolith/zstep/remote.py`: GPU/Docker/model preflight, source upload, screen and fresh verify execution.
- Create `src/evolith/zstep/loop.py`: candidate state machine and best-verified baseline advancement.
- Modify `src/evolith/cli.py`: `zstep-evolve`, `zstep-status`, `zstep-evidence`.
- Modify `.gitignore`: exclude `targets/zstep/`; existing `*.db` rules cover SQLite state.
- Create focused tests under `tests/test_zstep_*.py`; use fake proposers/runners for loop tests.
- Modify `README.md`: document the isolated source-level workflow and remote permission prerequisite.

### Task 1: Task contract and benchmark types

**Files:**
- Create: `src/evolith/zstep/__init__.py`
- Create: `src/evolith/zstep/task.py`
- Test: `tests/test_zstep_task.py`

- [ ] **Step 1: Write failing contract tests**

```python
from pathlib import Path
import pytest
from pydantic import ValidationError
from evolith.zstep.task import ZStepRSITask


def make_task(tmp_path: Path) -> ZStepRSITask:
    return ZStepRSITask(
        project_root=tmp_path,
        source_repo=tmp_path / "original-zstep",
        baseline_sha="596d4ef",
    )


def test_task_keeps_all_source_inside_evolith(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    assert task.upstream_dir == tmp_path / "targets/zstep/upstream"
    assert task.candidates_dir == tmp_path / "targets/zstep/candidates"
    assert task.state_dir == tmp_path / ".evolith/zstep-rsi"
    assert task.max_generations == 1


def test_task_rejects_overlapping_public_and_hidden_tiers(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="disjoint"):
        ZStepRSITask(
            project_root=tmp_path,
            source_repo=tmp_path / "original-zstep",
            baseline_sha="596d4ef",
            screen_tiers="in512_c1:1:8:512:64",
            verify_tiers="in512_c1:1:8:512:256",
        )
```

- [ ] **Step 2: Run the tests and confirm the import fails**

Run: `uv run pytest tests/test_zstep_task.py -v`
Expected: FAIL because `evolith.zstep.task` does not exist.

- [ ] **Step 3: Implement the contract and metric models**

Create an empty `src/evolith/zstep/__init__.py`, then define:

```python
from __future__ import annotations

from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, model_validator


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    LOCAL_REJECTED = "local_rejected"
    SCREEN_REJECTED = "screen_rejected"
    VERIFY_REJECTED = "verify_rejected"
    ACCEPTED = "accepted"
    FAILED = "failed"


class TierMetric(BaseModel):
    label: str
    repeat: int
    output_throughput_tok_s: float
    median_ttft_ms: float
    mean_tpot_ms: float
    p99_itl_ms: float


class VerificationDecision(BaseModel):
    accepted: bool
    geomean_gain: float
    reasons: list[str] = Field(default_factory=list)


class ZStepRSITask(BaseModel):
    project_root: Path
    source_repo: Path
    baseline_sha: str = "596d4ef"
    remote_user: str = "ubuntu"
    remote_host: str = "gpu-server"
    remote_root: str = "evolith-rsi"
    model_dir: str = "/data02/DeepSeek-V4.1-Flash-1017"
    model_id: str = "DeepSeek-V4.1-Flash"
    model_type: str = "deepseek_v41"
    image: str = "vllm/vllm-openai:v0.31.0"
    max_generations: int = Field(default=1, ge=1, le=100)
    max_changed_lines: int = Field(default=300, ge=1)
    local_timeout_sec: int = Field(default=600, ge=1)
    remote_timeout_sec: int = Field(default=7200, ge=60)
    repeats: int = Field(default=3, ge=1)
    min_geomean_gain: float = 0.03
    max_tier_throughput_regression: float = 0.03
    max_ttft_regression: float = 0.05
    max_tpot_regression: float = 0.05
    max_p99_itl_regression: float = 0.05
    screen_tiers: str = "in512_c1:1:8:512:64"
    verify_tiers: str = (
        "in512_c16:16:48:512:256;in512_c128:128:256:512:256;"
        "in1024_c32:32:200:1024:256;in1024_c128:128:400:1024:256"
    )
    allowed_roots: tuple[str, ...] = (
        "worker/zstep_worker/models/deepseek_v41.py",
        "worker/zstep_worker/runtime_loop.py",
        "worker/zstep_worker/kernels_sm120",
        "crates/zs-scheduler/src",
        "crates/zs-server/src/main.rs",
        "worker/tests",
    )

    @property
    def upstream_dir(self) -> Path:
        return self.project_root / "targets/zstep/upstream"

    @property
    def candidates_dir(self) -> Path:
        return self.project_root / "targets/zstep/candidates"

    @property
    def state_dir(self) -> Path:
        return self.project_root / ".evolith/zstep-rsi"

    @model_validator(mode="after")
    def validate_tiers(self) -> "ZStepRSITask":
        public = {item.split(":", 1)[0] for item in self.screen_tiers.split(";") if item}
        hidden = {item.split(":", 1)[0] for item in self.verify_tiers.split(";") if item}
        if not public or not hidden:
            raise ValueError("screen and verify tiers must be non-empty")
        if public & hidden:
            raise ValueError("screen and verify tier labels must be disjoint")
        return self
```

- [ ] **Step 4: Run contract tests**

Run: `uv run pytest tests/test_zstep_task.py -v`
Expected: 2 PASS.

### Task 2: Isolated workspaces and patch policy

**Files:**
- Create: `src/evolith/zstep/workspace.py`
- Test: `tests/test_zstep_workspace.py`

- [ ] **Step 1: Write failing workspace tests using a temporary git repository**

Create a repo with `git init`, one allowlisted file and one forbidden `Cargo.toml`. Configure a temporary author only if a test commit requires it. Test that the manager clones from the source without modifying it, applies an accepted patch to the next candidate, rejects a forbidden path, rejects more than 300 changed lines, and includes intent-to-add untracked files in `git diff --binary`.

Core assertions:

```python
def test_candidate_is_isolated_and_patch_stack_is_replayed(tmp_path: Path) -> None:
    source = make_source_repo(tmp_path)
    task = make_task(tmp_path, source)
    manager = ZStepWorkspace(task)
    manager.prepare_upstream()
    first = manager.create_candidate("g0001-test", [])
    write(first / "worker/zstep_worker/runtime_loop.py", "# candidate one\n")
    patch = manager.capture_patch(first)
    assert patch
    second = manager.create_candidate("g0002-test", [patch])
    assert read(second / "worker/zstep_worker/runtime_loop.py") == "# candidate one\n"
    assert read(source / "worker/zstep_worker/runtime_loop.py") == "# original\n"


def test_forbidden_manifest_change_is_rejected(tmp_path: Path) -> None:
    candidate = manager.create_candidate("g0001-forbidden", [])
    write(candidate / "Cargo.toml", "[workspace]\n")
    with pytest.raises(ValueError, match="forbidden path"):
        manager.capture_patch(candidate)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_zstep_workspace.py -v`
Expected: FAIL because `ZStepWorkspace` is missing.

- [ ] **Step 3: Implement the workspace manager**

Use argument-list subprocesses, never shell strings. Implement `ZStepWorkspace(task)` with `prepare_upstream()`, `create_candidate(candidate_id, accepted_patches)`, `capture_patch(candidate_dir)`, and `freeze_source(source_dir, destination) -> Path`. The constructor stores the task. `freeze_source` uses `tarfile` to write a gzipped archive containing only tracked source plus intent-to-add files, explicitly excluding `.git`, `target`, `__pycache__`, `.pytest_cache` and `.ruff_cache`; it returns `destination`. The other methods use these exact operations:

```python
def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def prepare_upstream(self) -> None:
    if self.task.upstream_dir.exists():
        return
    self.task.upstream_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--no-hardlinks", str(self.task.source_repo), str(self.task.upstream_dir)])
    _run(["git", "checkout", "--detach", self.task.baseline_sha], self.task.upstream_dir)


def create_candidate(self, candidate_id: str, accepted_patches: list[bytes]) -> Path:
    self.prepare_upstream()
    destination = self.task.candidates_dir / candidate_id
    if destination.exists():
        raise FileExistsError(destination)
    self.task.candidates_dir.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--no-hardlinks", str(self.task.upstream_dir), str(destination)])
    for patch in accepted_patches:
        subprocess.run(["git", "apply", "-"], cwd=destination, input=patch, check=True, capture_output=True)
    return destination
```

For `capture_patch`, run `git ls-files --others --exclude-standard`, reject any path outside `allowed_roots`, run `git add -N -- <untracked>` so new files enter the diff, inspect `git diff --numstat`, reject totals above `task.max_changed_lines`, and return `git diff --binary --full-index`. A path is allowed only when it equals an allowlisted file or is relative to an allowlisted directory. Raise on an empty diff.

- [ ] **Step 4: Run workspace tests**

Run: `uv run pytest tests/test_zstep_workspace.py -v`
Expected: all workspace tests PASS and the original source hash/diff remains unchanged.

### Task 3: SQLite lineage and accepted patch stack

**Files:**
- Create: `src/evolith/zstep/archive.py`
- Test: `tests/test_zstep_archive.py`

- [ ] **Step 1: Write failing persistence tests**

```python
def test_archive_restores_accepted_patch_stack(tmp_path: Path) -> None:
    archive = ZStepArchive(tmp_path / "archive.sqlite3")
    archive.start_candidate("g0001-a", 1, None, b"patch-a", "sha-a")
    archive.accept_candidate("g0001-a", "evidence/g0001-a", 0.04)
    archive.start_candidate("g0002-b", 2, "g0001-a", b"patch-b", "sha-b")
    archive.finish_candidate("g0002-b", CandidateStatus.VERIFY_REJECTED, "evidence/g0002-b", "latency")
    assert archive.accepted_patches() == [b"patch-a"]
    assert archive.current_parent() == "g0001-a"
    assert archive.next_generation() == 3
    restored = ZStepArchive(tmp_path / "archive.sqlite3")
    assert restored.accepted_patches() == [b"patch-a"]
```

Also test status listing order and evidence-path retrieval.

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_zstep_archive.py -v`
Expected: FAIL because `ZStepArchive` is missing.

- [ ] **Step 3: Implement the archive**

Create tables:

```sql
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL,
    parent_id TEXT,
    patch BLOB NOT NULL,
    source_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    evidence_dir TEXT NOT NULL DEFAULT '',
    failure_reason TEXT NOT NULL DEFAULT '',
    geomean_gain REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

Implement `start_candidate`, `finish_candidate`, `accept_candidate`, `accepted_patches`, `current_parent`, `next_generation`, `list_candidates`, and `evidence_dir`. `accepted_patches()` selects patches with `status='accepted'` ordered by generation; `current_parent()` returns the latest accepted ID or `None`; `next_generation()` is `COALESCE(MAX(generation), 0) + 1`. Close connections in `close()` and support context-manager use.

- [ ] **Step 4: Run archive tests**

Run: `uv run pytest tests/test_zstep_archive.py -v`
Expected: all archive tests PASS, including reopening the database.

### Task 4: Kimi proposer and local static gates

**Files:**
- Create: `src/evolith/zstep/proposer.py`
- Test: `tests/test_zstep_proposer.py`

- [ ] **Step 1: Write failing proposer tests with a fake executable**

Create a fake `kimi` script that writes its current directory, arguments and whether `SSHPASS` exists to a JSON file, then edits an allowlisted source file. Assert that `KimiCodeProposer.propose()` calls `kimi -p <prompt>` in the candidate directory, returns stdout, writes an evidence log, and removes `SSHPASS`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and similar credential variables from the child environment. Also test timeout and nonzero exit errors.

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_zstep_proposer.py -v`
Expected: FAIL because the proposer is missing.

- [ ] **Step 3: Implement the proposer and local command runner**

```python
class KimiCodeProposer:
    def __init__(self, task: ZStepRSITask, executable: str = "kimi") -> None:
        self.task = task
        self.executable = executable

    def propose(self, candidate_dir: Path, prompt: str, evidence_dir: Path) -> str:
        env = {k: v for k, v in os.environ.items() if not _is_credential(k)}
        result = subprocess.run(
            [self.executable, "-p", prompt],
            cwd=candidate_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=self.task.local_timeout_sec,
        )
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "proposer.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Kimi proposer exited {result.returncode}")
        return result.stdout
```

`_is_credential` must perform case-insensitive matching for exact names and suffixes including `SSHPASS`, `_PASSWORD`, `_TOKEN`, `_API_KEY`, and `_SECRET`; preserve ordinary paths and Kimi configuration needed to run. Add `run_local_gates(candidate_dir, task, evidence_dir)` in `gates.py`: always run `cargo fmt --all --check` and `cargo test --workspace` on Linux (on macOS, test the portable `zs-kv` and `zs-wire` packages because `zs-shm` requires Linux `eventfd`); when `ruff` is available run `ruff check worker/`; save each command's combined log and stop on the first nonzero exit. Do not install missing tools automatically.

- [ ] **Step 4: Run proposer tests**

Run: `uv run pytest tests/test_zstep_proposer.py -v`
Expected: all proposer and environment-scrubbing tests PASS.

### Task 5: Benchmark parsing and acceptance gates

**Files:**
- Create: `src/evolith/zstep/gates.py`
- Test: `tests/test_zstep_gates.py`

- [ ] **Step 1: Write failing parser and decision tests**

Use small inline JSON documents with two labels and three repeats. Cover:

```python
def test_acceptance_requires_gain_and_all_guards() -> None:
    baseline = metrics(throughput=100.0, ttft=100.0, tpot=10.0, itl=20.0)
    candidate = metrics(throughput=105.0, ttft=102.0, tpot=10.2, itl=20.5)
    decision = evaluate_candidate(baseline, candidate, parity_report(0, 1.0), default_task())
    assert decision.accepted
    assert decision.geomean_gain == pytest.approx(0.05)


def test_hidden_tier_regression_rejects_candidate() -> None:
    decision = evaluate_candidate(
        metrics(throughput=100.0), metrics(throughput=95.0),
        parity_report(0, 1.0), default_task(),
    )
    assert not decision.accepted
    assert any("throughput" in reason for reason in decision.reasons)


def test_recompilation_or_prime_square_mismatch_rejects_candidate() -> None:
    assert not evaluate_candidate(good_base(), good_candidate(), parity_report(1, 1.0), default_task()).accepted
    assert not evaluate_candidate(good_base(), good_candidate(), parity_report(0, 0.99), default_task()).accepted
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_zstep_gates.py -v`
Expected: FAIL because parser/evaluator functions are missing.

- [ ] **Step 3: Implement parsers and deterministic acceptance**

Implement:

```python
def load_benchmark(path: Path) -> dict[str, list[TierMetric]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[TierMetric]] = {}
    for row in payload["tiers"]:
        metric = TierMetric.model_validate(row)
        grouped.setdefault(metric.label, []).append(metric)
    return grouped


def load_parity(path: Path) -> tuple[int, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prime = next(case for case in payload["cases"] if case["name"] == "prime_squares_table")
    rates = {
        mode: float(prime[f"{mode}_match"]["exact_token_match_rate"])
        for mode in ("single", "batched", "chunked")
    }
    return int(payload["triton_recompilations"]), rates
```

The parity evaluator must require exactly the `single`, `batched` and `chunked` rates and every rate must equal `1.0`. Aggregate each benchmark label by median across repeats. Compute geomean gain as `exp(mean(log(candidate/base))) - 1`. Reject when gain `< task.min_geomean_gain`, any tier throughput regression exceeds its limit, any latency regression exceeds its limit, recompilations are nonzero, or prime-square coverage/rate is incomplete. Return every failed reason in `VerificationDecision`.

- [ ] **Step 4: Run gate tests**

Run: `uv run pytest tests/test_zstep_gates.py -v`
Expected: all parser and acceptance tests PASS.

### Task 6: Remote preflight and isolated execution

**Files:**
- Create: `src/evolith/zstep/remote.py`
- Test: `tests/test_zstep_remote.py`

- [ ] **Step 1: Write failing command-construction tests with fake subprocess**

Assert that:

- the SSH destination is `ubuntu@gpu-server`;
- preflight runs `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` and rejects any row whose memory is nonzero;
- `docker info` permission denial becomes a clear `RemotePreflightError` and does not invoke sudo;
- screen and verify roots differ and include the candidate ID;
- rsync excludes `.git`, `target`, `__pycache__` and caches;
- the zstep command uses the task's exact screen or verify tier string, `--repeats`, `--model-dir`, `--allreduce-blocks 8`, and output paths;
- no constructed command contains `sudo`, `docker rm`, or an unscoped process-kill command.

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_zstep_remote.py -v`
Expected: FAIL because the remote runner is missing.

- [ ] **Step 3: Implement `ZStepRemoteRunner`**

Implement `ZStepRemoteRunner(task)` with `preflight()`, `screen(candidate_id, source, evidence) -> Path`, and `verify(candidate_id, baseline_archive, candidate_archive, evidence) -> tuple[Path, Path, Path]`. The constructor stores the task and builds SSH argument lists without embedding credentials.

Use a `_ssh(args, timeout)` wrapper around `ssh -o BatchMode=yes` when no password helper is configured, and support the existing `SSHPASS` environment through `sshpass -e ssh` without logging the password. Preflight must check eight GPUs, Docker access, the container model `config.json` model type, and create a candidate-specific remote lock with atomic `mkdir`; stale locks are not automatically deleted.

Package the candidate with `tar` and transfer with rsync or scp. On the remote host, prepare the SM120 overlay with zstep's existing `scripts/prepare_sm120_overlay.sh`, build `zs-server`, run the documented greedy audit against `docs/benchmarks/phase3_vllm_greedy_reference.json`, and run the tier benchmark. `screen()` returns the local metrics JSON path. `verify()` must re-extract both frozen archives into fresh verify subdirectories, then run baseline and candidate in round-major order where supported; otherwise explicitly record sequential fresh runs as a limitation rather than claiming interleaving. It returns candidate metrics, baseline metrics and parity JSON paths.

Download stdout/stderr and all JSON artifacts to the candidate evidence directory. Cleanup only candidate-owned paths and processes. Do not use sudo, global `/dev/shm` deletion, or pre-existing remote source directories.

- [ ] **Step 4: Run remote unit tests and a real preflight**

Run: `uv run pytest tests/test_zstep_remote.py -v`
Expected: all unit tests PASS.

Then invoke only `preflight()` through the new CLI or a short Python call.
Expected at present: eight GPUs free, followed by explicit failure that `ubuntu` cannot access `/var/run/docker.sock`; no benchmark starts and sudo is not used.

### Task 7: RSI state machine

**Files:**
- Create: `src/evolith/zstep/loop.py`
- Test: `tests/test_zstep_loop.py`

- [ ] **Step 1: Write failing fake-driven state-machine tests**

Define fake proposer/workspace/gates/runner doubles and cover:

1. local rejection does not call screen;
2. screen rejection does not call verify;
3. verify rejection preserves the accepted patch stack;
4. acceptance appends exactly one patch and the next candidate receives it;
5. source hash changed between screen and verify raises and rejects;
6. remote preflight failure becomes a recorded failed candidate without a retry loop;
7. restarting with the same archive starts at the next generation.

The acceptance test should assert:

```python
assert statuses == [CandidateStatus.ACCEPTED, CandidateStatus.VERIFY_REJECTED]
assert archive.accepted_patches() == [first_patch]
assert proposer.prompts[1].startswith("Parent: g0001")
assert "latency" in proposer.prompts[1]
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_zstep_loop.py -v`
Expected: FAIL because `ZStepRSILoop` is missing.

- [ ] **Step 3: Implement `ZStepRSILoop.run()`**

For each generation up to `task.max_generations`:

```python
candidate_id = f"g{generation:04d}-{uuid.uuid4().hex[:8]}"
parent_id = archive.current_parent()
patches = archive.accepted_patches()
candidate_dir = workspace.create_candidate(candidate_id, patches)
evidence_dir = task.state_dir / "evidence" / candidate_id
evidence_dir.mkdir(parents=True, exist_ok=False)
baseline_archive = workspace.freeze_source(candidate_dir, evidence_dir / "baseline.tar.gz")
proposer.propose(candidate_dir, prompt, evidence_dir)
patch = workspace.capture_patch(candidate_dir)
patch_path = task.state_dir / "patches" / f"{candidate_id}.patch"
patch_path.parent.mkdir(parents=True, exist_ok=True)
patch_path.write_bytes(patch)
candidate_archive = workspace.freeze_source(candidate_dir, evidence_dir / "candidate.tar.gz")
source_hash = hashlib.sha256(candidate_archive.read_bytes()).hexdigest()
archive.start_candidate(candidate_id, generation, parent_id, patch, source_hash)
run_local_gates(candidate_dir, task, evidence_dir)
remote.preflight()
remote.screen(candidate_id, candidate_archive, evidence_dir)
candidate_json, baseline_json, parity_json = remote.verify(
    candidate_id, baseline_archive, candidate_archive, evidence_dir
)
decision = evaluate_candidate(
    load_benchmark(baseline_json), load_benchmark(candidate_json), parity_json, task
)
if decision.accepted:
    archive.accept_candidate(candidate_id, str(evidence_dir), decision.geomean_gain)
else:
    archive.finish_candidate(
        candidate_id, CandidateStatus.VERIFY_REJECTED, str(evidence_dir), "; ".join(decision.reasons)
    )
```

Wrap each stage with explicit exception-to-status mapping and preserve the exception text in evidence. Create the archive row immediately after a nonempty validated patch exists. For failures before patch creation, write an evidence `failure.json` without fabricating a lineage row. Build the next prompt from parent ID, accepted patch summaries, compressed prior status and public screen metrics; never include `task.verify_tiers` or hidden seeds in the proposer prompt. Save both frozen source archives and the candidate SHA-256 before screening. Update the accepted baseline only after `VerificationDecision.accepted` is true.

- [ ] **Step 4: Run state-machine tests**

Run: `uv run pytest tests/test_zstep_loop.py -v`
Expected: all fake-driven acceptance, rejection, isolation and recovery tests PASS.

### Task 8: CLI, ignore rules and user documentation

**Files:**
- Modify: `src/evolith/cli.py`
- Modify: `.gitignore`
- Modify: `README.md`
- Test: `tests/test_zstep_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Use Typer's `CliRunner`. Monkeypatch `ZStepRSILoop` and assert `zstep-evolve --generations 2` constructs a task rooted at the current Evolith directory and invokes one loop with two generations. Test `zstep-status` on an empty archive and `zstep-evidence <id>` for an existing and unknown candidate. Assert default source is `~/github/zstep`, default generation count is 1, and help states that no deployment or upstream commit occurs.

- [ ] **Step 2: Run CLI tests and confirm failure**

Run: `uv run pytest tests/test_zstep_cli.py -v`
Expected: FAIL because the commands are missing.

- [ ] **Step 3: Add commands and documentation**

Add:

```text
evolith zstep-evolve --source ~/github/zstep --host gpu-server --user ubuntu --generations 1
evolith zstep-status
evolith zstep-evidence g0001-<suffix>
```

Resolve paths before constructing `ZStepRSITask`; keep passwords out of options and logs. Add `/targets/zstep/` to `.gitignore`. README must explain independent source copies, Kimi CLI requirement, Docker permission prerequisite, real verification gates, evidence location, default one generation, and the fact that accepted patches are experimental baselines rather than deployed code.

- [ ] **Step 4: Run CLI tests**

Run: `uv run pytest tests/test_zstep_cli.py -v`
Expected: all CLI tests PASS.

### Task 9: Full verification and real-scenario attempt

**Files:**
- Verify all created/modified files; no new scaffolding beyond this plan.

- [ ] **Step 1: Run the complete Evolith test suite**

Run: `uv run pytest -v`
Expected: all old and new tests PASS.

- [ ] **Step 2: Run build/package validation**

Run: `uv build`
Expected: wheel and sdist build successfully and include `evolith.zstep` modules, but exclude `targets/zstep` and `.evolith` artifacts.

- [ ] **Step 3: Check workspace hygiene**

Run: `git status --short`
Expected: only intentional source, test, documentation and `.gitignore` changes. Remove build outputs created solely by verification if they are not desired deliverables; never remove pre-existing user files.

- [ ] **Step 4: Exercise the real command end to end**

Run `uv run evolith zstep-evolve --generations 1` with the existing SSH credential mechanism. The command must create Evolith-owned source copies and invoke a real Kimi candidate before remote preflight.

If Docker remains inaccessible, record the explicit preflight failure and preserve the candidate/evidence. Do not claim real GPU verification passed. If access has been fixed, continue through smoke, fresh verify and accept/reject; confirm eight GPUs return to 0 MiB and only candidate-owned remote paths remain.

- [ ] **Step 5: Report verified and blocked results separately**

Report test/build results, candidate ID, evidence path and exact accept/reject reason. If remote Docker permission blocks execution, state that the loop is implemented and locally tested but real zstep GPU verification remains unverified; request Docker access rather than using sudo or weakening the verifier.
