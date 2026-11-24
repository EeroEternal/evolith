"""Local Git-versioned Memory Ring loader for Evolith RSI loops.

Reads human/scout directives, scoped subsystem hardware facts, and stepping-stone
queues directly from ``<project_root>/memory/`` with zero network latency, keeping
static subsystem knowledge deterministic for Automatic Prefix Caching (APC).
"""

from __future__ import annotations

from pathlib import Path
import re

_ROOTS_COMMENT_RE = re.compile(r"^\s*<!--\s*roots\s*:\s*(.*?)\s*-->\s*$")


def _normalize_rel(path_str: str) -> str:
    return path_str.strip().strip("/")


def _roots_overlap(card_roots: tuple[str, ...], allowed_roots: tuple[str, ...]) -> bool:
    if not card_roots or not allowed_roots:
        return True
    norm_allowed = tuple(_normalize_rel(r) for r in allowed_roots if r.strip())
    if not norm_allowed:
        return True
    for c_root in card_roots:
        c_norm = _normalize_rel(c_root)
        if not c_norm:
            continue
        for a_norm in norm_allowed:
            if (
                a_norm == c_norm
                or a_norm.startswith(f"{c_norm}/")
                or c_norm.startswith(f"{a_norm}/")
            ):
                return True
    return False


def _parse_fact_card(path: Path) -> tuple[tuple[str, ...], str]:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    declared_roots: tuple[str, ...] = ()
    body_lines: list[str] = []
    for idx, line in enumerate(raw_lines):
        if idx < 3:
            match = _ROOTS_COMMENT_RE.match(line)
            if match:
                declared_roots = tuple(
                    item.strip() for item in match.group(1).split(",") if item.strip()
                )
                continue
        body_lines.append(line)
    return declared_roots, "\n".join(body_lines).strip()


def _extract_markdown_section(path: Path, heading_prefix: str) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    captured: list[str] = []
    in_section = False
    target = heading_prefix.strip().lower()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if in_section:
                break
            if stripped[3:].strip().lower().startswith(target):
                in_section = True
            continue
        if in_section:
            if stripped.startswith(">"):
                continue
            captured.append(line)
    return "\n".join(captured).strip()


def load_memory_ring(
    project_root: Path,
    allowed_roots: tuple[str, ...] = (),
    *,
    max_chars: int = 10000,
) -> str:
    """Load core playbook laws, scoped facts, stepping stones, and directives from ``memory/``."""
    memory_dir = project_root / "memory"
    if not memory_dir.is_dir() or max_chars <= 0:
        return ""

    sections: list[str] = []

    # 0. Universal 10-law inference optimization digest (100% static across all campaigns -> ideal APC prefix)
    playbook_digest = _extract_markdown_section(
        memory_dir / "playbook.md", "core laws"
    )
    if playbook_digest:
        sections.append(
            "[Full-Lifecycle Inference Optimization Laws (`memory/playbook.md`)]\n"
            + playbook_digest
        )

    # 1. Static subsystem fact cards (maximizes APC prefix cache reuse within a campaign)
    facts_dir = memory_dir / "facts"
    if facts_dir.is_dir():
        matched_cards: list[str] = []
        for fact_file in sorted(facts_dir.glob("*.md")):
            if not fact_file.is_file():
                continue
            card_roots, card_body = _parse_fact_card(fact_file)
            if card_body and _roots_overlap(card_roots, allowed_roots):
                matched_cards.append(card_body)
        if matched_cards:
            sections.append(
                "[Compounding Hardware & Engine Facts (`memory/facts/`)]\n"
                + "\n\n".join(matched_cards)
            )

    # 2. Near-miss positive stepping stones for orthogonal composition
    stepping_body = _extract_markdown_section(
        memory_dir / "stepping-stones.md", "2."
    ) or _extract_markdown_section(
        memory_dir / "stepping-stones.md", "near-miss"
    )
    if stepping_body:
        sections.append(
            "[Orthogonal Stepping-Stones Queue (`memory/stepping-stones.md`)]\n"
            + stepping_body
        )

    # 3. Active human / scout directives
    directives_body = _extract_markdown_section(
        memory_dir / "directives.md", "active directives"
    )
    if directives_body:
        sections.append(
            "[Active Human & Scout Directives (`memory/directives.md`)]\n"
            + directives_body
        )

    if not sections:
        return ""

    combined = "\n\n".join(sections)
    if len(combined) > max_chars:
        combined = combined[: max_chars - 24].rstrip() + "\n...[truncated]"
    return combined
