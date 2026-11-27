"""Declarative Target Surface, Static Dependency Closure & Action Ladder Planner.

Brings RSInfer-style impact analysis (`unit.toml` + import graph + Action Ladder)
to Evolith's outer loop and inner `workbench` engine:
1. Maps any `git diff` or `IdentityDiff` onto a deterministic 7-rung Action Ladder
   (`RUNG_0_SIM_ONLY` .. `RUNG_6_ENVELOPE_GAP`), computing affected units,
   transitive downstream consumers, CUDA Graph recapture requirements, and
   shared state resource touches (`--dry-run` deterministic).
2. Upgrades `composer.py` stepping-stone composition from naive file-disjointness
   to transitive dependency + shared state/resource closure orthogonality.
"""

from __future__ import annotations

import ast
from enum import IntEnum
from pathlib import Path, PurePosixPath
import re

from pydantic import BaseModel


class ConfigBindingMode(str):
    LIVE = "live"
    RECAPTURE_GRAPH = "recapture_graph"
    RELOAD_UNIT = "reload_unit"
    RESTART = "restart"


class ActionRung(IntEnum):
    """Ordered minimal-safe action ladder (higher rung = heavier action)."""

    RUNG_0_SIM_ONLY = 0
    RUNG_1_LIVE_CONFIG = 1
    RUNG_2_RELOAD_LOGIC = 2
    RUNG_3_HOTSWAP_NATIVE_OP = 3
    RUNG_4_REFIT_STATE = 4
    RUNG_5_RESTART_SCHEDULER = 5
    RUNG_6_ENVELOPE_GAP = 6


class ConfigKeyDecl(BaseModel):
    key: str
    owner_unit: str
    binding: str = ConfigBindingMode.RELOAD_UNIT


class StateResourceDecl(BaseModel):
    name: str
    budget_bytes: int | None = None
    description: str = ""
    patch_patterns: tuple[str, ...] = ()


class UnitDecl(BaseModel):
    unit_id: str
    kind: str = "logic"  # "logic" | "state" | "native_op" | "scheduler"
    path_prefixes: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    captures_cuda_graph: bool = False
    requires_drain: bool = True
    hotswap_extern_c: bool = False
    shared_resources: tuple[str, ...] = ()
    refit_patterns: tuple[str, ...] = ()


class TargetSurface(BaseModel):
    """Declarative boundary manifest for a target inference engine."""

    target_name: str
    pinned_paths: tuple[str, ...] = ()
    allowed_roots: tuple[str, ...] = ()
    units: tuple[UnitDecl, ...] = ()
    config_keys: tuple[ConfigKeyDecl, ...] = ()
    state_resources: tuple[StateResourceDecl, ...] = ()


class BlastRadiusPlan(BaseModel):
    """Deterministic `--dry-run` impact report for a candidate ChangeSet."""

    rung: ActionRung
    rung_name: str
    directly_touched_units: tuple[str, ...] = ()
    reload_closure_units: tuple[str, ...] = ()
    recapture_cuda_graphs: bool = False
    requires_drain: bool = False
    touched_resources: tuple[str, ...] = ()
    pinned_violations: tuple[str, ...] = ()
    forbidden_root_violations: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def can_hot_apply_in_workbench(self) -> bool:
        return (
            not self.pinned_violations
            and not self.forbidden_root_violations
            and self.rung <= ActionRung.RUNG_4_REFIT_STATE
        )


def _matches_prefix(rel_path: str, prefix: str) -> bool:
    p = PurePosixPath(rel_path)
    pref = PurePosixPath(prefix)
    return p == pref or pref in p.parents


def _match_most_specific_unit(rel_path: str, units: tuple[UnitDecl, ...]) -> UnitDecl | None:
    best: UnitDecl | None = None
    best_len = -1
    for u in units:
        for pref in u.path_prefixes:
            if _matches_prefix(rel_path, pref) and len(pref) > best_len:
                best = u
                best_len = len(pref)
    return best


def build_python_import_reverse_graph(
    source_root: Path | None,
    python_pkg_root: str = "worker",
) -> dict[str, set[str]]:
    """Build `imported_module -> set(importer_rel_paths)` across a Python package root."""
    rev: dict[str, set[str]] = {}
    if source_root is None:
        return rev
    pkg_dir = source_root / python_pkg_root
    if not pkg_dir.is_dir():
        return rev

    mod_to_rel: dict[str, str] = {}
    for py_file in sorted(pkg_dir.rglob("*.py")):
        rel_to_pkg = py_file.relative_to(pkg_dir)
        parts = list(rel_to_pkg.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            mod_to_rel[".".join(parts)] = py_file.relative_to(source_root).as_posix()

    for py_file in sorted(pkg_dir.rglob("*.py")):
        importer_rel = py_file.relative_to(source_root).as_posix()
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                targets.append(node.module)
                for alias in node.names:
                    targets.append(f"{node.module}.{alias.name}")
            for mod in targets:
                if mod in mod_to_rel and mod_to_rel[mod] != importer_rel:
                    rev.setdefault(mod_to_rel[mod], set()).add(importer_rel)
    return rev


def _extract_patch_touched_resources(
    patch_text: str,
    surface: TargetSurface | None = None,
) -> set[str]:
    """Match declarative `StateResourceDecl.patch_patterns` from `surface` against a patch."""
    resources: set[str] = set()
    if not patch_text or surface is None:
        return resources
    for res in surface.state_resources:
        for pattern in res.patch_patterns:
            if re.search(pattern, patch_text):
                resources.add(res.name)
                break
    return resources


def analyze_changeset_blast_radius(
    surface: TargetSurface,
    *,
    changed_files: tuple[str, ...] = (),
    changed_config_keys: tuple[str, ...] = (),
    patch_text: str = "",
    source_root: Path | None = None,
) -> BlastRadiusPlan:
    """Derive minimal safe `ActionRung` and transitive reload closure for a ChangeSet."""
    notes: list[str] = []
    pinned_hits = tuple(
        sorted(f for f in changed_files if any(_matches_prefix(f, p) for p in surface.pinned_paths))
    )
    forbidden_hits: list[str] = []
    if surface.allowed_roots:
        for f in sorted(set(changed_files)):
            if not any(_matches_prefix(f, r) for r in surface.allowed_roots):
                forbidden_hits.append(f)

    if pinned_hits or forbidden_hits:
        notes.append("Change touches coordinator-pinned or non-allowlisted paths")
        return BlastRadiusPlan(
            rung=ActionRung.RUNG_6_ENVELOPE_GAP,
            rung_name=ActionRung.RUNG_6_ENVELOPE_GAP.name,
            pinned_violations=pinned_hits,
            forbidden_root_violations=tuple(forbidden_hits),
            notes=tuple(notes),
        )

    unit_by_id = {u.unit_id: u for u in surface.units}
    rev_deps: dict[str, set[str]] = {u.unit_id: set() for u in surface.units}
    for u in surface.units:
        for dep in u.depends_on:
            rev_deps.setdefault(dep, set()).add(u.unit_id)

    # Augment unit dependency closure with static Python import graph when source_root is provided
    py_rev = build_python_import_reverse_graph(source_root)
    expanded_files = set(changed_files)
    frontier = list(changed_files)
    while frontier:
        cur = frontier.pop()
        for importer in py_rev.get(cur, ()):
            if importer not in expanded_files:
                expanded_files.add(importer)
                frontier.append(importer)

    direct_units: set[str] = set()
    closure_units: set[str] = set()
    for f in changed_files:
        owner = _match_most_specific_unit(f, surface.units)
        if owner is not None:
            direct_units.add(owner.unit_id)
    for f in expanded_files:
        owner = _match_most_specific_unit(f, surface.units)
        if owner is not None:
            closure_units.add(owner.unit_id)

    # Propagate transitive unit reverse-dependencies
    u_queue = list(closure_units)
    while u_queue:
        uid = u_queue.pop()
        for consumer in rev_deps.get(uid, ()):
            if consumer not in closure_units:
                closure_units.add(consumer)
                u_queue.append(consumer)

    cfg_decl_map = {c.key: c for c in surface.config_keys}
    rung = ActionRung.RUNG_0_SIM_ONLY
    recapture_graphs = False
    requires_drain = False
    touched_resources = _extract_patch_touched_resources(patch_text, surface)

    for key in changed_config_keys:
        decl = cfg_decl_map.get(key)
        if decl is None:
            rung = max(rung, ActionRung.RUNG_5_RESTART_SCHEDULER)
            notes.append(f"Undeclared config key `{key}` modified; escalating conservatively")
        else:
            closure_units.add(decl.owner_unit)
            if decl.binding == ConfigBindingMode.LIVE:
                rung = max(rung, ActionRung.RUNG_1_LIVE_CONFIG)
            elif decl.binding == ConfigBindingMode.RECAPTURE_GRAPH:
                rung = max(rung, ActionRung.RUNG_2_RELOAD_LOGIC)
                recapture_graphs = True
                requires_drain = True
            elif decl.binding == ConfigBindingMode.RELOAD_UNIT:
                rung = max(rung, ActionRung.RUNG_2_RELOAD_LOGIC)
                requires_drain = True
            else:
                rung = max(rung, ActionRung.RUNG_5_RESTART_SCHEDULER)
                requires_drain = True

    for f in changed_files:
        matched_any = any(
            any(_matches_prefix(f, p) for p in u.path_prefixes) for u in surface.units
        )
        if not matched_any:
            rung = max(rung, ActionRung.RUNG_5_RESTART_SCHEDULER)
            requires_drain = True
            notes.append(f"Unmapped file `{f}` escalated conservatively to RUNG_5")

    for uid in sorted(closure_units):
        u = unit_by_id.get(uid)
        if u is None:
            continue
        touched_resources.update(u.shared_resources)
        if u.captures_cuda_graph:
            recapture_graphs = True
        if u.requires_drain:
            requires_drain = True

    for uid in sorted(direct_units):
        u = unit_by_id[uid]
        if u.kind == "scheduler":
            rung = max(rung, ActionRung.RUNG_5_RESTART_SCHEDULER)
            notes.append(f"Rust scheduler unit `{uid}` changed: StateKeeper can stay resident via SHM re-attach")
        elif u.kind == "state":
            rung = max(rung, ActionRung.RUNG_4_REFIT_STATE)
        elif u.kind == "native_op":
            if u.hotswap_extern_c:
                rung = max(rung, ActionRung.RUNG_3_HOTSWAP_NATIVE_OP)
                notes.append(f"Native op `{uid}` eligible for content-hashed extern-C .so hot-swap")
            else:
                rung = max(rung, ActionRung.RUNG_6_ENVELOPE_GAP)
                notes.append(f"Native op `{uid}` lacks extern-C hot-swap ABI")
        else:
            rung = max(rung, ActionRung.RUNG_2_RELOAD_LOGIC)
            if patch_text and any(re.search(pat, patch_text) for pat in u.refit_patterns):
                rung = max(rung, ActionRung.RUNG_4_REFIT_STATE)
                notes.append(f"Unit `{uid}` modified weight packing/loading hook -> RUNG_4 selective refit")

    return BlastRadiusPlan(
        rung=rung,
        rung_name=rung.name,
        directly_touched_units=tuple(sorted(direct_units)),
        reload_closure_units=tuple(sorted(closure_units)),
        recapture_cuda_graphs=recapture_graphs,
        requires_drain=requires_drain,
        touched_resources=tuple(sorted(touched_resources)),
        notes=tuple(notes),
    )


def are_patches_closure_orthogonal(
    surface: TargetSurface,
    *,
    left_files: tuple[str, ...],
    right_files: tuple[str, ...],
    left_patch_text: str = "",
    right_patch_text: str = "",
    source_root: Path | None = None,
) -> tuple[bool, str]:
    """Verify two patches have disjoint files, direct units, and shared state resources."""
    overlap_files = sorted(set(left_files) & set(right_files))
    if overlap_files:
        return False, f"Overlapping modified files: {overlap_files}"

    left_plan = analyze_changeset_blast_radius(
        surface,
        changed_files=left_files,
        patch_text=left_patch_text,
        source_root=source_root,
    )
    right_plan = analyze_changeset_blast_radius(
        surface,
        changed_files=right_files,
        patch_text=right_patch_text,
        source_root=source_root,
    )
    if left_plan.pinned_violations or right_plan.pinned_violations:
        return False, "One or both patches violate pinned contract files"

    overlap_direct = sorted(
        set(left_plan.directly_touched_units) & set(right_plan.directly_touched_units)
    )
    if overlap_direct:
        return False, f"Both patches mutate the same logical unit(s): {overlap_direct}"

    # Extract explicit patch-level shared state resource mutations (e.g. both growing ring budget)
    left_explicit_res = _extract_patch_touched_resources(left_patch_text, surface)
    right_explicit_res = _extract_patch_touched_resources(right_patch_text, surface)
    overlap_res = sorted(left_explicit_res & right_explicit_res)
    if overlap_res:
        return False, f"Implicit conflict on shared state resource(s): {overlap_res}"

    return True, "Orthogonal across files, logical units, and shared state resources"


def default_zstep_surface() -> TargetSurface:
    """Canonical declarative surface manifest for `zstep` (Rust + SM120 Python/CUDA)."""
    return TargetSurface(
        target_name="zstep",
        pinned_paths=(
            "crates/zs-server/src/lib.rs",
            "crates/zs-wire/src/lib.rs",
            "docs/benchmarks/phase3_vllm_greedy_reference.json",
            "scripts/prepare_sm120_overlay.sh",
            "worker/tests/test_shm_layout_parity.py",
            "worker/zstep_worker/shm_bridge.py",
        ),
        allowed_roots=("crates", "worker/zstep_worker"),
        units=(
            UnitDecl(
                unit_id="zs.scheduler",
                kind="scheduler",
                path_prefixes=("crates/zs-scheduler", "crates/zs-kv", "crates/zs-shm", "crates/zs-server"),
                requires_drain=True,
                shared_resources=("kv_swa_page_pool",),
            ),
            UnitDecl(
                unit_id="kernels.shm_allreduce_csrc",
                kind="native_op",
                path_prefixes=("worker/zstep_worker/kernels_sm120/csrc",),
                captures_cuda_graph=True,
                requires_drain=True,
                hotswap_extern_c=True,
                shared_resources=("pxb_ring_budget",),
            ),
            UnitDecl(
                unit_id="kernels.shm_allreduce_py",
                kind="logic",
                path_prefixes=("worker/zstep_worker/kernels_sm120/shm_allreduce.py",),
                depends_on=("kernels.shm_allreduce_csrc",),
                captures_cuda_graph=True,
                requires_drain=True,
                shared_resources=("pxb_ring_budget",),
            ),
            UnitDecl(
                unit_id="kernels.b12x_sm120",
                kind="logic",
                path_prefixes=("worker/zstep_worker/kernels_sm120/b12x_dsv41.py",),
                depends_on=("kernels.shm_allreduce_py",),
                captures_cuda_graph=True,
                requires_drain=True,
                refit_patterns=(r"\bpack_weight\b", r"\bprocess_weights_after_loading\b"),
            ),
            UnitDecl(
                unit_id="models.runner",
                kind="logic",
                path_prefixes=("worker/zstep_worker/models", "worker/zstep_worker/spec_decode.py"),
                depends_on=("kernels.b12x_sm120", "kernels.shm_allreduce_py"),
                captures_cuda_graph=True,
                requires_drain=True,
            ),
            UnitDecl(
                unit_id="worker.runtime",
                kind="logic",
                path_prefixes=("worker/zstep_worker",),
                depends_on=("models.runner",),
                captures_cuda_graph=True,
                requires_drain=True,
            ),
        ),
        config_keys=(
            ConfigKeyDecl(
                key="sched.max_mixed_step_tokens",
                owner_unit="zs.scheduler",
                binding=ConfigBindingMode.LIVE,
            ),
            ConfigKeyDecl(
                key="sched.max_prefill_chunk",
                owner_unit="zs.scheduler",
                binding=ConfigBindingMode.LIVE,
            ),
            ConfigKeyDecl(
                key="worker.decode_graph_buckets",
                owner_unit="models.runner",
                binding=ConfigBindingMode.RECAPTURE_GRAPH,
            ),
        ),
        state_resources=(
            StateResourceDecl(
                name="pxb_ring_budget",
                budget_bytes=256 * 1024 * 1024,
                description="PP=1 static PXB IPC ring ceiling (256 MiB)",
                patch_patterns=(
                    r"\bHostMappedShmAllReduce\b|\bpxb_max_elements\b|\benable_pxb_p2p\b",
                ),
            ),
            StateResourceDecl(
                name="mhc_overlap_state",
                description="SM120 mHC stream/allreduce overlap binding table",
                patch_patterns=(
                    r"\battach_sm120_mhc_shm_ar\b|\b_prepare_sm120_mhc_overlap\b|\bshm_ar_dec",
                ),
            ),
            StateResourceDecl(
                name="packed_weights",
                description="GPU-resident MXFP4/MXFP8 packed linear & expert weights",
                patch_patterns=(
                    r"\bpack_weight\b|\bprocess_weights_after_loading\b|\b_b12x_wo_a_packed\b",
                ),
            ),
            StateResourceDecl(
                name="cuda_graph_pool",
                description="Captured decode & piecewise CUDA graph pool",
                patch_patterns=(r"\bCUDAGraph\b|\bcapture_decode_graphs\b|\bpiecewise\b",),
            ),
            StateResourceDecl(
                name="kv_swa_page_pool",
                description="Physical paged KV, MLA, SWA, and Indexer buffers",
                patch_patterns=(
                    r"\bswa_page_pool\b|\bnum_kv_pages\b|\bkv_leases\b|\bprefix_lease\b",
                ),
            ),
        ),
    )
