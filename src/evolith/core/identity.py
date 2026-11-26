"""Deterministic SystemIdentity, structured diff, and immutable snapshot staging.

Unifies provenance across Evolith adapters (`lab`, `zstep`, `workbench`) by
hashing `(files, config, env_allowlist, artifacts, topology, platform)` into a
single canonical digest and staging read-only snapshots so candidate workers
never import from a mutable agent workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform as py_platform
import shutil
from typing import Any

from pydantic import BaseModel, Field

_DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".evolith",
        "target",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".venv",
    }
)


class SystemIdentity(BaseModel):
    """Canonical cryptographic identity for a target engine snapshot + runtime state."""

    digest: str
    files: dict[str, str] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    env_allowlist: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    topology: dict[str, Any] = Field(default_factory=dict)
    platform: dict[str, str] = Field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.digest[:16]


class IdentityDiff(BaseModel):
    """Structured difference between two `SystemIdentity` manifests."""

    identical: bool
    added_files: tuple[str, ...] = ()
    removed_files: tuple[str, ...] = ()
    modified_files: tuple[str, ...] = ()
    changed_config_keys: tuple[str, ...] = ()
    changed_env_keys: tuple[str, ...] = ()
    changed_artifacts: tuple[str, ...] = ()
    topology_changed: bool = False
    platform_changed: bool = False

    @property
    def all_changed_files(self) -> tuple[str, ...]:
        return tuple(sorted({*self.added_files, *self.removed_files, *self.modified_files}))


def _hash_file_entry(path: Path) -> str:
    if path.is_symlink():
        target = str(path.readlink()).encode("utf-8", errors="surrogateescape")
        return "symlink:" + hashlib.sha256(target).hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def fingerprint_model_manifest(model_dir: Path | None) -> dict[str, str]:
    """Lightweight deterministic model checkpoint manifest (config/index/shard sizes).

    Matches Evolith's documented boundary in README: covers config, index, and
    shard byte sizes rather than hashing multi-hundred-GiB weight contents.
    """
    if model_dir is None or not model_dir.is_dir():
        return {}
    manifest: dict[str, str] = {}
    for entry in sorted(model_dir.iterdir(), key=lambda p: p.name):
        if entry.name.startswith("."):
            continue
        if entry.is_file():
            if entry.suffix in {".json", ".tiktoken", ".model"} and entry.stat().st_size <= 8_388_608:
                manifest[entry.name] = _hash_file_entry(entry)
            else:
                manifest[entry.name] = f"size:{entry.stat().st_size}"
    return manifest


def compute_system_identity(
    source_root: Path,
    *,
    include_roots: tuple[str, ...] = (),
    config: dict[str, Any] | None = None,
    allowed_env_keys: tuple[str, ...] = (),
    environ: dict[str, str] | None = None,
    artifacts: dict[str, str] | None = None,
    model_dir: Path | None = None,
    topology: dict[str, Any] | None = None,
    platform_info: dict[str, str] | None = None,
) -> SystemIdentity:
    """Compute a deterministic `SystemIdentity` over source files and declared context."""
    root = source_root.resolve()
    files: dict[str, str] = {}
    scan_targets: list[Path]
    if include_roots:
        scan_targets = [root / rel for rel in sorted(set(include_roots)) if (root / rel).exists()]
    else:
        scan_targets = [root]

    for target in scan_targets:
        if target.is_file() or target.is_symlink():
            rel = target.relative_to(root).as_posix()
            files[rel] = _hash_file_entry(target)
            continue
        for path in sorted(target.rglob("*")):
            rel_parts = path.relative_to(root).parts
            if _DEFAULT_EXCLUDED_DIRS & set(rel_parts):
                continue
            if path.is_file() or path.is_symlink():
                rel = path.relative_to(root).as_posix()
                files[rel] = _hash_file_entry(path)

    env_src = environ if environ is not None else dict(os.environ)
    env_snap = {
        key: str(env_src[key])
        for key in sorted(set(allowed_env_keys))
        if key in env_src
    }
    merged_artifacts = dict(artifacts or {})
    for k, v in fingerprint_model_manifest(model_dir).items():
        merged_artifacts[f"model/{k}"] = v

    plat = dict(
        platform_info
        or {
            "system": py_platform.system(),
            "machine": py_platform.machine(),
            "python_major_minor": f"{py_platform.python_version_tuple()[0]}.{py_platform.python_version_tuple()[1]}",
        }
    )
    cfg = dict(config or {})
    topo = dict(topology or {})

    canonical = {
        "schema": "evolith.identity.v1",
        "files": files,
        "config": cfg,
        "env_allowlist": env_snap,
        "artifacts": merged_artifacts,
        "topology": topo,
        "platform": plat,
    }
    digest = hashlib.sha256(_canonical_json_bytes(canonical)).hexdigest()
    return SystemIdentity(
        digest=digest,
        files=files,
        config=cfg,
        env_allowlist=env_snap,
        artifacts=merged_artifacts,
        topology=topo,
        platform=plat,
    )


def diff_identities(base: SystemIdentity, candidate: SystemIdentity) -> IdentityDiff:
    """Return a deterministic structural diff between two `SystemIdentity` objects."""
    base_paths = set(base.files)
    cand_paths = set(candidate.files)
    added = tuple(sorted(cand_paths - base_paths))
    removed = tuple(sorted(base_paths - cand_paths))
    modified = tuple(
        sorted(p for p in (base_paths & cand_paths) if base.files[p] != candidate.files[p])
    )
    cfg_keys = tuple(
        sorted(
            k
            for k in (set(base.config) | set(candidate.config))
            if base.config.get(k) != candidate.config.get(k)
        )
    )
    env_keys = tuple(
        sorted(
            k
            for k in (set(base.env_allowlist) | set(candidate.env_allowlist))
            if base.env_allowlist.get(k) != candidate.env_allowlist.get(k)
        )
    )
    art_keys = tuple(
        sorted(
            k
            for k in (set(base.artifacts) | set(candidate.artifacts))
            if base.artifacts.get(k) != candidate.artifacts.get(k)
        )
    )
    topo_changed = base.topology != candidate.topology
    plat_changed = base.platform != candidate.platform
    identical = (
        base.digest == candidate.digest
        and not added
        and not removed
        and not modified
        and not cfg_keys
        and not env_keys
        and not art_keys
        and not topo_changed
        and not plat_changed
    )
    return IdentityDiff(
        identical=identical,
        added_files=added,
        removed_files=removed,
        modified_files=modified,
        changed_config_keys=cfg_keys,
        changed_env_keys=env_keys,
        changed_artifacts=art_keys,
        topology_changed=topo_changed,
        platform_changed=plat_changed,
    )


def materialize_readonly_snapshot(
    source_root: Path,
    snapshot_store_dir: Path,
    identity: SystemIdentity | None = None,
) -> tuple[Path, SystemIdentity]:
    """Stage an immutable read-only snapshot directory keyed by `SystemIdentity.digest`.

    Ensures live worker processes import Python/CUDA modules from a frozen
    content-addressed tree rather than an actively edited workspace.
    """
    ident = identity or compute_system_identity(source_root)
    dest = snapshot_store_dir / ident.short_id
    marker = dest / ".evolith-identity.json"
    if marker.is_file():
        recorded = SystemIdentity.model_validate_json(marker.read_text(encoding="utf-8"))
        if recorded.digest == ident.digest:
            return dest, ident

    staging = snapshot_store_dir / f".staging-{ident.short_id}-{os.getpid()}"
    if staging.exists():
        _make_tree_writable(staging)
        shutil.rmtree(staging, ignore_errors=True)
    snapshot_store_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_root,
        staging,
        symlinks=True,
        ignore=shutil.ignore_patterns(*sorted(_DEFAULT_EXCLUDED_DIRS)),
    )
    (staging / ".evolith-identity.json").write_text(
        ident.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    _freeze_tree_permissions(staging)
    if dest.exists():
        _make_tree_writable(dest)
        shutil.rmtree(dest)
    os.replace(staging, dest)
    return dest, ident


def _freeze_tree_permissions(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        dp = Path(dirpath)
        for fname in filenames:
            fp = dp / fname
            if fp.is_symlink():
                continue
            mode = fp.stat().st_mode
            fp.chmod(0o555 if (mode & 0o111) else 0o444)
        dp.chmod(0o555)


def _make_tree_writable(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root, topdown=True):
        dp = Path(dirpath)
        dp.chmod(0o755)
        for fname in filenames:
            fp = dp / fname
            if fp.is_symlink():
                continue
            fp.chmod(0o644)
