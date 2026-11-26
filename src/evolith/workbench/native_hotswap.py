"""Content-Hashed `extern "C"` C/C++/CUDA Shared Library Hot-Swapper & Preflight.

Overcomes Python/CUDA `.so` unload limitations without restarting a 475 GiB
resident worker process:
1. Hashes source + compiler + architecture flags into a unique shared object
   filename `libevo_<op>_<sha12>.so` (written atomically via temp file + rename
   so each mutation gets an isolated `dlopen` handle across all 8 GPU ranks).
2. Supports both host C (`cc`/`clang`/`gcc`) and CUDA (`nvcc -shared -Xcompiler
   -fPIC -gencode=arch=compute_120,code=sm_120 -lcudart`) with pure scalar/pointer
   `extern "C"` ABIs (`~1s` compile time, zero `<torch/extension.h>` overhead).
3. Runs mandatory out-of-process preflight smoke checks BEFORE any resident
   `StateKeeper` rank calls `dlopen`, ensuring GPU `asm("trap;");`, illegal memory
   accesses, missing symbols, segfaults, or infinite spin-loops die inside a
   disposable subprocess rather than poisoning the 8-GPU CUDA context.
"""

from __future__ import annotations

from collections.abc import Sequence
import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


class NativeOpPreflightError(RuntimeError):
    """Raised when a candidate C/C++/CUDA `.so` fails compilation or subprocess preflight."""


_CTYPE_MAP: dict[str, Any] = {
    "i32": ctypes.c_int32,
    "int32": ctypes.c_int32,
    "int": ctypes.c_int,
    "u32": ctypes.c_uint32,
    "uint32": ctypes.c_uint32,
    "i64": ctypes.c_int64,
    "int64": ctypes.c_int64,
    "u64": ctypes.c_uint64,
    "uint64": ctypes.c_uint64,
    "usize": ctypes.c_size_t,
    "f32": ctypes.c_float,
    "float": ctypes.c_float,
    "f64": ctypes.c_double,
    "double": ctypes.c_double,
    "ptr": ctypes.c_void_p,
    "void_p": ctypes.c_void_p,
    "void": None,
    "none": None,
}


def resolve_ctype_spec(spec: Any) -> Any:
    """Resolve a string ABI token (e.g. `'u64'`, `'i32'`, `'void'`) or passthrough `ctypes` type."""
    if spec is None:
        return None
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key not in _CTYPE_MAP:
            raise ValueError(f"Unsupported ctype ABI token `{spec}`; valid={sorted(_CTYPE_MAP)}")
        return _CTYPE_MAP[key]
    return spec


@dataclass(frozen=True)
class CompiledNativeOp:
    op_name: str
    symbol_name: str
    content_hash: str
    so_path: Path
    backend: str = "c"  # "c" | "cuda"


class ExternCOpLoader:
    """Compiles, subprocess-verifies, and dynamically loads `extern "C"` C & CUDA operators."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        compiler: str | None = None,
        nvcc_compiler: str | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compiler = compiler or self._detect_host_compiler()
        self.nvcc_compiler = nvcc_compiler or self._detect_nvcc_compiler()
        self._loaded_libs: dict[str, ctypes.CDLL] = {}

    @staticmethod
    def _detect_host_compiler() -> str:
        for candidate in ("cc", "clang", "gcc", "c++"):
            found = shutil.which(candidate)
            if found:
                return found
        return ""

    @staticmethod
    def _detect_nvcc_compiler() -> str:
        found = shutil.which("nvcc")
        if found:
            return found
        fallback = Path("/usr/local/cuda/bin/nvcc")
        return str(fallback) if fallback.is_file() else ""

    @staticmethod
    def detect_cuda_arch_flag(arch: str | None = None) -> str:
        """Format `-gencode=arch=compute_XX,code=sm_XX` (auto-detecting via `torch.cuda` if omitted)."""
        if arch:
            cleaned = arch.strip().lower().removeprefix("sm_").removeprefix("compute_")
            return f"-gencode=arch=compute_{cleaned},code=sm_{cleaned}"
        try:
            import torch

            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                code = f"{major}{minor}"
                return f"-gencode=arch=compute_{code},code=sm_{code}"
        except Exception:
            pass
        return "-gencode=arch=compute_120,code=sm_120"

    def compile_c_source(
        self,
        op_name: str,
        c_source: str,
        *,
        symbol_name: str,
        extra_cflags: tuple[str, ...] = ("-O2",),
        timeout_sec: float = 20.0,
    ) -> CompiledNativeOp:
        """Compile `extern "C"` C source to a unique content-addressed shared library."""
        if not self.compiler:
            raise NativeOpPreflightError("No C/C++ compiler found on PATH")
        digest_input = f"c|{self.compiler}|{extra_cflags}|{symbol_name}|\n{c_source}".encode("utf-8")
        sha12 = hashlib.sha256(digest_input).hexdigest()[:12]
        ext = ".dylib" if platform.system() == "Darwin" else ".so"
        out_so = self.cache_dir / f"libevo_{op_name}_{sha12}{ext}"
        src_path = self.cache_dir / f"evo_{op_name}_{sha12}.c"

        if not out_so.is_file():
            src_path.write_text(c_source, encoding="utf-8")
            tmp_so = self.cache_dir / f".tmp_libevo_{op_name}_{sha12}_{os.getpid()}{ext}"
            shared_flags = (
                ["-dynamiclib", "-fPIC"]
                if platform.system() == "Darwin"
                else ["-shared", "-fPIC"]
            )
            cmd = [self.compiler, *shared_flags, *extra_cflags, str(src_path), "-o", str(tmp_so)]
            self._run_compiler(cmd, op_name=op_name, tmp_so=tmp_so, out_so=out_so, timeout_sec=timeout_sec)

        return CompiledNativeOp(
            op_name=op_name,
            symbol_name=symbol_name,
            content_hash=sha12,
            so_path=out_so,
            backend="c",
        )

    def compile_cuda_source(
        self,
        op_name: str,
        cuda_source: str,
        *,
        symbol_name: str,
        arch: str | None = None,
        extra_nvcc_flags: tuple[str, ...] = ("-O3",),
        timeout_sec: float = 45.0,
    ) -> CompiledNativeOp:
        """Compile `extern "C"` CUDA `.cu` source via `nvcc` into `libevo_<op>_<sha12>.so`."""
        if not self.nvcc_compiler:
            raise NativeOpPreflightError("`nvcc` compiler not found on PATH or /usr/local/cuda/bin/nvcc")
        arch_flag = self.detect_cuda_arch_flag(arch)
        digest_input = (
            f"cuda|{self.nvcc_compiler}|{arch_flag}|{extra_nvcc_flags}|{symbol_name}|\n{cuda_source}"
        ).encode("utf-8")
        sha12 = hashlib.sha256(digest_input).hexdigest()[:12]
        out_so = self.cache_dir / f"libevo_{op_name}_{sha12}.so"
        src_path = self.cache_dir / f"evo_{op_name}_{sha12}.cu"

        if not out_so.is_file():
            src_path.write_text(cuda_source, encoding="utf-8")
            tmp_so = self.cache_dir / f".tmp_libevo_{op_name}_{sha12}_{os.getpid()}.so"
            cmd = [
                self.nvcc_compiler,
                "-shared",
                "-Xcompiler",
                "-fPIC",
                arch_flag,
                *extra_nvcc_flags,
                str(src_path),
                "-lcudart",
                "-o",
                str(tmp_so),
            ]
            self._run_compiler(cmd, op_name=op_name, tmp_so=tmp_so, out_so=out_so, timeout_sec=timeout_sec)

        return CompiledNativeOp(
            op_name=op_name,
            symbol_name=symbol_name,
            content_hash=sha12,
            so_path=out_so,
            backend="cuda",
        )

    @staticmethod
    def _run_compiler(
        cmd: list[str],
        *,
        op_name: str,
        tmp_so: Path,
        out_so: Path,
        timeout_sec: float,
    ) -> None:
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            tmp_so.unlink(missing_ok=True)
            raise NativeOpPreflightError(
                f"Compiler timed out after {timeout_sec}s on `{op_name}`"
            ) from exc
        if res.returncode != 0:
            tmp_so.unlink(missing_ok=True)
            raise NativeOpPreflightError(
                f"Compilation failed for `{op_name}`: {(res.stderr or res.stdout).strip()[:400]}"
            )
        os.replace(tmp_so, out_so)

    def preflight_in_subprocess(
        self,
        compiled: CompiledNativeOp,
        *,
        smoke_expr: str = "",
        timeout_sec: float = 10.0,
        cuda_device: str | None = None,
    ) -> None:
        """Verify `dlopen`, symbol presence, and optional CPU/GPU smoke script in a child process."""
        script = (
            "import ctypes, sys\n"
            f"lib = ctypes.CDLL({str(compiled.so_path)!r})\n"
            f"fn = getattr(lib, {compiled.symbol_name!r})\n"
        )
        if smoke_expr:
            script += smoke_expr + "\n"

        env = os.environ.copy()
        if cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        try:
            res = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise NativeOpPreflightError(
                f"Subprocess preflight timed out after {timeout_sec}s (possible hang in `{compiled.op_name}`)"
            ) from exc
        if res.returncode != 0:
            detail = (res.stderr or res.stdout or f"signal {-res.returncode}").strip()
            raise NativeOpPreflightError(
                f"Subprocess preflight crashed/rejected `{compiled.op_name}` "
                f"(exit={res.returncode}): {detail[-400:]}"
            )

    def bind_symbol_in_process(
        self,
        compiled: CompiledNativeOp,
        *,
        argtypes: Sequence[Any] = (),
        restype: Any = ctypes.c_int,
    ) -> Any:
        """Load an already-preflighted `.so` into the current resident rank process."""
        key = f"{compiled.op_name}:{compiled.content_hash}"
        lib = self._loaded_libs.get(key)
        if lib is None:
            lib = ctypes.CDLL(str(compiled.so_path))
            self._loaded_libs[key] = lib
        fn = getattr(lib, compiled.symbol_name)
        fn.argtypes = [resolve_ctype_spec(a) for a in argtypes]
        fn.restype = resolve_ctype_spec(restype)
        return fn

    def load_verified_symbol(
        self,
        compiled: CompiledNativeOp,
        *,
        argtypes: Sequence[Any] = (),
        restype: Any = ctypes.c_int,
        smoke_expr: str = "",
        timeout_sec: float = 10.0,
        cuda_device: str | None = None,
    ) -> Any:
        """Run mandatory subprocess preflight, then load into the resident process."""
        self.preflight_in_subprocess(
            compiled,
            smoke_expr=smoke_expr,
            timeout_sec=timeout_sec,
            cuda_device=cuda_device,
        )
        return self.bind_symbol_in_process(compiled, argtypes=argtypes, restype=restype)
