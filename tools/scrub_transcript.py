#!/usr/bin/env python3
"""Scrub secrets, host IPs, usernames, and home paths from session transcripts.

Supports Antigravity ``transcript.jsonl`` logs, Claude Code ``.jsonl`` session
files, and plain text/Markdown files. Exits with code 1 (fail-closed) when any
high-entropy mixed-case+digit token remains that could be an unredacted secret.
Standard library only.
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence

CLIP_LIMIT = 2000

_SECRETS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    ("docker token", re.compile(r"\bdckr_(?:pat|oat)_[A-Za-z0-9_-]{16,}")),
    ("hugging face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    (
        "github token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    ),
    ("api key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("aws key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "bearer token",
        re.compile(r"(?i)\b(bearer|authorization:)\s+[A-Za-z0-9._~+/=-]{16,}"),
    ),
    ("url password", re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)")),
    (
        "signed url",
        re.compile(
            r"(?i)\b(X-(?:Amz|Goog)-(?:Signature|Credential|Security-Token)|Signature|sig)=[A-Za-z0-9%._~+/-]{16,}"
        ),
    ),
    (
        "assigned secret",
        re.compile(
            r"(?i)\b([A-Z0-9_]*(?:SSHPASS|TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY))(\s*[=:]\s*)(['\"]?)([^\s'\"]{4,})\3"
        ),
    ),
)
_BLOB_RE = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
_PRIVATE: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<email>"),
    ("home directory", re.compile(r"(/Users|/home)/[A-Za-z0-9._-]+"), r"\1/<user>"),
    (
        "ip address",
        re.compile(r"\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)(?:\d{1,3}\.){3}\d{1,3}\b"),
        "<ip>",
    ),
)
_CANDIDATE_RE = re.compile(r"[A-Za-z0-9+/_=-]{32,}")
_HEX_RE = re.compile(r"[0-9a-fA-F]+")


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = {ch: text.count(ch) for ch in set(text)}
    length = len(text)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def looks_like_secret(token: str) -> bool:
    if "/" in token:
        return False
    for part in re.split(r"[-_=]", token):
        if len(part) < 24 or _HEX_RE.fullmatch(part):
            continue
        classes = sum(bool(re.search(pat, part)) for pat in (r"[a-z]", r"[A-Z]", r"[0-9]"))
        if classes == 3 and shannon_entropy(part) >= 4.3:
            return True
    return False


def scrub_text(
    text: str,
    *,
    allow_prefixes: Sequence[str] = (),
    redact_names: Sequence[str] = (),
) -> tuple[str, dict[str, int], list[tuple[int, str]]]:
    counts: dict[str, int] = {}

    def _counted(kind: str, repl: str):
        def _sub(match: re.Match[str]) -> str:
            counts[kind] = counts.get(kind, 0) + 1
            return match.expand(repl) if "\\" in repl else repl

        return _sub

    keep_templates = {
        "assigned secret": r"\1\2[REDACTED]",
        "bearer token": r"\1 [REDACTED]",
        "signed url": r"\1=[REDACTED]",
    }
    for kind, pattern in _SECRETS:
        text = pattern.sub(_counted(kind, keep_templates.get(kind, "[REDACTED]")), text)
    text = _BLOB_RE.sub(_counted("data blob", "[DATA]"), text)
    for kind, pattern, repl in _PRIVATE:
        text = pattern.sub(_counted(kind, repl), text)

    for name in sorted(set(redact_names), key=len, reverse=True):
        if len(name) >= 3:
            pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])")
            text = pat.sub(_counted("name", "<redacted>"), text)

    suspects: list[tuple[int, str]] = []
    seen: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        for cand in _CANDIDATE_RE.findall(line):
            if (
                cand not in seen
                and looks_like_secret(cand)
                and not any(cand.startswith(prefix) for prefix in allow_prefixes)
            ):
                seen.add(cand)
                suspects.append((lineno, cand))
    return text, counts, suspects


def _clip(val: object) -> str:
    txt = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
    if len(txt) <= CLIP_LIMIT:
        return txt
    return txt[:CLIP_LIMIT] + f"\n... [{len(txt) - CLIP_LIMIT} more characters]"


def _iter_claude_blocks(content: object) -> Iterable[str]:
    if isinstance(content, str):
        yield content
        return
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            yield str(block.get("text") or "")
        elif btype == "tool_use":
            inp = block.get("input") or {}
            body = (
                inp.get("command")
                or inp.get("file_path")
                or json.dumps(inp, ensure_ascii=False)
            )
            yield f"**Tool: {block.get('name')}**\n\n```\n{_clip(body)}\n```"
        elif btype == "tool_result":
            res = block.get("content")
            if isinstance(res, list):
                res = "\n".join(str(b.get("text") or "") for b in res if isinstance(b, dict))
            yield f"**Result**\n\n```\n{_clip(res or '')}\n```"


def render_session(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".jsonl":
        return raw
    out: list[str] = [f"# Session transcript ({path.stem})\n"]
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        # Antigravity transcript.jsonl format (`type` in USER_INPUT / PLANNER_RESPONSE)
        step_type = row.get("type")
        if step_type in ("USER_INPUT", "PLANNER_RESPONSE"):
            pieces: list[str] = []
            content = str(row.get("content") or "").strip()
            if content:
                pieces.append(_clip(content))
            for call in row.get("tool_calls") or []:
                if isinstance(call, dict):
                    name = call.get("name") or call.get("tool_name") or "tool"
                    args = call.get("arguments") or call.get("args") or {}
                    pieces.append(f"**Tool: {name}**\n\n```\n{_clip(args)}\n```")
            if pieces:
                heading = "User" if step_type == "USER_INPUT" else "Assistant"
                out.append(f"## {heading}\n\n" + "\n\n".join(pieces) + "\n")
            continue
        # Claude Code JSONL format
        if step_type in ("user", "assistant") and not row.get("isSidechain"):
            msg = row.get("message")
            parts = [
                p
                for p in _iter_claude_blocks(msg.get("content") if isinstance(msg, dict) else None)
                if p.strip()
            ]
            if parts:
                out.append(f"## {str(step_type).capitalize()}\n\n" + "\n\n".join(parts) + "\n")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path, help="Path to .jsonl transcript or text file")
    parser.add_argument("-o", "--output", type=Path, default=Path("transcript.md"))
    parser.add_argument(
        "--redact",
        action="append",
        default=[],
        metavar="NAME",
        help="Additional account, host, or organization string to redact (repeatable)",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="PREFIX",
        help="Prefix of a flagged high-entropy token confirmed safe by the owner",
    )
    parser.add_argument(
        "--keep-login",
        action="store_true",
        help="Do not automatically redact the current OS login username",
    )
    args = parser.parse_args(argv)
    names = list(args.redact)
    if not args.keep_login:
        try:
            names.append(getpass.getuser())
        except Exception:
            pass
    scrubbed, counts, suspects = scrub_text(
        render_session(args.session),
        allow_prefixes=tuple(args.allow),
        redact_names=names,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(scrubbed, encoding="utf-8")
    summary = ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "nothing"
    print(f"wrote {args.output} ({len(scrubbed)} chars); replaced: {summary}", flush=True)
    if suspects:
        print(
            f"NOT READY TO PUBLISH: {len(suspects)} suspected secret(s) remain in {args.output}:",
            file=sys.stderr,
        )
        for lineno, item in suspects[:20]:
            print(f"  line {lineno}: {item[:16]}... ({len(item)} chars)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
