"""Fail-closed verification of the diagnostic compiler policy, not admission."""

import ast
import hashlib
from pathlib import Path

_SHAPE_PRESERVING = {"broadcast_to", "reshape", "expand_dims", "view", "permute", "trans"}
_REDUCTIONS = {"sum", "max", "min", "reduce", "argmax", "argmin", "cumsum", "cumprod",
               "xor_sum", "associative_scan", "histogram", "sort", "prod", "flip"}


def compiler_policy_report(rank: int, requested: str | None, actual: bool) -> dict:
    if type(rank) is not int or rank not in range(4):
        raise ValueError("Compiler policy requires rank 0–3.")
    if requested not in ("0", "1") or type(actual) is not bool:
        raise ValueError("Explicit, supported compiler determinism policy is required.")
    expected = requested == "1"
    if actual != expected:
        raise ValueError("Compiler ignored the requested determinism policy.")
    return {"rank": rank, "requested": expected, "inductor_deterministic": actual,
            "scope": "compiler reduction policy only; not whole-model correctness proof"}


def _dtype_class(node) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "tl":
        if node.attr == "int1":
            return "bool"
        if node.attr.startswith(("int", "uint")):
            return "int"
        if node.attr.startswith(("float", "bfloat")):
            return "float"
    return None


def _pointer_class(kind: str) -> str | None:
    if kind == "*i1":
        return "ptr:bool"
    if kind.startswith(("*i", "*u")):
        return "ptr:int"
    if kind.startswith(("*fp", "*bf")):
        return "ptr:float"
    return None


def _join(*kinds):
    if None in kinds or any(k.startswith("ptr") for k in kinds):
        return None
    if "float" in kinds:
        return "float"
    return "int" if "int" in kinds else "bool"


class _IntegerReductionProof:
    """Conservative dtype propagation over one generated Triton kernel.

    Integer addition is associative even when it wraps, so the reduction order
    cannot change an integer sum. Anything not provably integer is unproven.
    """

    def __init__(self, function: ast.FunctionDef, signature: dict):
        self.env = {}
        for arg in function.args.args:
            kind = signature.get(arg.arg)
            if not isinstance(kind, str):
                continue
            if kind.startswith("*"):
                self.env[arg.arg] = _pointer_class(kind)
            elif kind.startswith(("i", "u")) or (
                kind == "constexpr" and arg.arg.endswith(("BLOCK", "numel"))
            ):
                self.env[arg.arg] = "int"
        self.reductions = []
        self.unsupported = False
        self.block(function.body)

    def kind(self, node):
        if isinstance(node, ast.Constant):
            value = node.value
            return ("bool" if type(value) is bool else "int" if type(value) is int
                    else "float" if type(value) is float else None)
        if isinstance(node, ast.Name):
            return self.env.get(node.id)
        if isinstance(node, ast.Subscript):
            return self.kind(node.value)
        if isinstance(node, ast.Attribute) and node.attr == "dtype":
            return self.kind(node.value)
        if isinstance(node, (ast.Compare, ast.BoolOp)) or (
            isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
        ):
            return "bool"
        if isinstance(node, ast.UnaryOp):
            return self.kind(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = self.kind(node.left), self.kind(node.right)
            if None in (left, right) or isinstance(node.op, (ast.Pow, ast.MatMult)):
                return None
            if isinstance(node.op, ast.Div):
                return "float"
            if left.startswith("ptr") or right.startswith("ptr"):
                if isinstance(node.op, ast.Add) and {left, right} & {"int"} and left != right:
                    return left if left.startswith("ptr") else right
                return None
            if left == right == "bool" and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
                return "bool"
            return _join(left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name, owner = node.func.attr, node.func.value
            if isinstance(owner, ast.Name) and owner.id == "tl":
                if name == "load" and node.args:
                    pointer = self.kind(node.args[0])
                    return pointer.split(":")[1] if pointer and pointer.startswith("ptr:") else None
                if name == "full":
                    dtype = node.args[2] if len(node.args) > 2 else next(
                        (kw.value for kw in node.keywords if kw.arg == "dtype"), None)
                    return _dtype_class(dtype) or (self.kind(dtype) if dtype is not None else None)
                if name in ("arange", "program_id", "num_programs"):
                    return "int"
                if name == "where" and len(node.args) == 3:
                    return _join(self.kind(node.args[1]), self.kind(node.args[2])) if None not in (
                        self.kind(node.args[1]), self.kind(node.args[2])) else None
                if name in _SHAPE_PRESERVING and node.args:
                    return self.kind(node.args[0])
                if name == "sum" and node.args:
                    operand = self.kind(node.args[0])
                    return "int" if operand in ("int", "bool") else operand
                return None
            if name == "to" and node.args:
                return _dtype_class(node.args[0])
            if name in _SHAPE_PRESERVING:
                return self.kind(owner)
        return None

    def scan(self, node):
        for call in ast.walk(node):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)):
                continue
            owner, name = call.func.value.id, call.func.attr
            if owner == "triton_helpers" or (owner == "tl" and (
                    name in _REDUCTIONS or name.startswith("atomic"))):
                operand = self.kind(call.args[0]) if call.args else None
                self.reductions.append((owner, name, operand))

    def block(self, statements):
        for statement in statements:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(
                    statement.targets[0], ast.Name):
                self.scan(statement.value)
                self.env[statement.targets[0].id] = self.kind(statement.value)
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) \
                    and statement.value is not None:
                self.scan(statement.value)
                self.env[statement.target.id] = self.kind(statement.value)
            elif isinstance(statement, ast.Expr):
                self.scan(statement.value)
            elif (isinstance(statement, ast.For) and isinstance(statement.target, ast.Name)
                  and isinstance(statement.iter, ast.Call)
                  and isinstance(statement.iter.func, ast.Name)
                  and statement.iter.func.id == "range" and not statement.orelse):
                self.env[statement.target.id] = "int"
                before = dict(self.env)
                self.block(statement.body)
                first = dict(self.env)
                self.block(statement.body)
                for name in set(first) | set(self.env) | set(before):
                    if len({before.get(name, first.get(name)), first.get(name),
                            self.env.get(name)}) > 1:
                        self.env[name] = None
            else:
                self.unsupported = True

    @property
    def proven(self) -> bool:
        return (not self.unsupported and bool(self.reductions)
                and all(owner == "tl" and name == "sum" and operand in ("int", "bool")
                        for owner, name, operand in self.reductions))


def _signature(decorator: ast.Call) -> dict:
    meta = next((kw.value for kw in decorator.keywords if kw.arg == "triton_meta"), None)
    if not isinstance(meta, ast.Dict):
        return {}
    for key, value in zip(meta.keys, meta.values):
        if isinstance(key, ast.Constant) and key.value == "signature":
            try:
                parsed = ast.literal_eval(value)
            except ValueError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def integer_only_reduction(function: ast.FunctionDef, decorator: ast.Call) -> bool:
    """True only when every reduction provably sums integer or Boolean values."""
    return _IntegerReductionProof(function, _signature(decorator)).proven


def reduction_policy_inventory(root: Path, expected: bool) -> list[dict]:
    """Verify generated metadata as well as each worker's import-time setting.

    Order-sensitive (floating or unproven) reductions must carry the requested
    policy. Integer-only sums are recorded but cannot satisfy the check alone.
    """
    if type(expected) is not bool:
        raise ValueError("Explicit compiler policy is required.")
    records = []
    for path in root.rglob("*.py"):
        source = path.read_text()
        if "@triton_heuristics." not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if not (isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Attribute)
                        and isinstance(decorator.func.value, ast.Name)
                        and decorator.func.value.id == "triton_heuristics"
                        and decorator.func.attr in ("reduction", "persistent_reduction")):
                    continue
                metadata = next((kw.value for kw in decorator.keywords
                                 if kw.arg == "inductor_meta"), None)
                flag = None
                if isinstance(metadata, ast.Dict):
                    flag = next((value.value for key, value in zip(metadata.keys, metadata.values)
                                 if isinstance(key, ast.Constant) and key.value == "deterministic"
                                 and isinstance(value, ast.Constant)), None)
                integer_only = integer_only_reduction(node, decorator)
                if type(flag) is not bool or (flag is not expected and not integer_only):
                    raise ValueError(
                        "Generated reduction did not retain the requested compiler policy: "
                        f"{path.relative_to(root)}:{node.name}."
                    )
                records.append({"path": str(path.relative_to(root)), "kernel": node.name,
                                "deterministic": flag, "integer_only": integer_only,
                                "source_sha256": hashlib.sha256(source.encode()).hexdigest()})
    if not any(not row["integer_only"] for row in records):
        raise ValueError("No order-sensitive generated reduction available to verify compiler policy.")
    return records
