"""Layer 1 of the sandbox: static AST safety analysis (runs BEFORE any code executes).

Allowlist for imports (anything not listed is rejected), denylist for dangerous
builtins / attributes / dunder introspection. Static analysis is cheap and gives
precise line numbers for feedback, but it can be evaded by dynamic tricks, which is
why the runtime audit hook (sandbox_boot.py) and OS limits exist as later layers.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

ALLOWED_MODULES = {
    # stdlib (pure computation / formatting)
    "math", "statistics", "json", "csv", "datetime", "time", "random", "collections", "itertools",
    "functools", "operator", "re", "string", "textwrap", "typing", "dataclasses", "enum", "pathlib",
    "os", "sys", "warnings", "copy", "decimal", "fractions", "numbers", "heapq", "bisect", "calendar",
    "io", "pprint", "__future__",
    # scientific stack
    "numpy", "pandas", "matplotlib", "scipy", "sklearn",
}
BLOCKED_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "input", "globals", "memoryview"}
# attributes that are dangerous on the os / sys modules (resolved through import aliases)
BLOCKED_MODULE_ATTRS = {
    "os": {"system", "popen", "fork", "forkpty", "kill", "killpg", "startfile", "execl", "execle", "execlp",
           "execlpe", "execv", "execve", "execvp", "execvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe",
           "spawnv", "spawnve", "spawnvp", "spawnvpe", "posix_spawn", "posix_spawnp", "putenv", "unsetenv",
           "chmod", "chown", "setuid", "setgid", "symlink", "link", "_exit"},
    "sys": {"addaudithook", "settrace", "setprofile", "_getframe", "modules"},
}
# introspection attributes used for classic Python sandbox escapes
BLOCKED_DUNDERS = {"__subclasses__", "__globals__", "__builtins__", "__code__", "__closure__", "__bases__",
                   "__mro__", "__getattribute__", "__reduce__", "__reduce_ex__", "__import__", "__loader__",
                   "__spec__", "__self__", "__func__", "f_globals", "f_locals", "f_back", "gi_frame", "tb_frame"}
BLOCKED_ANY_ATTRS = {"read_pickle", "to_pickle"}  # unpickling = arbitrary code execution


@dataclass
class Violation:
    line: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"line {self.line}: [{self.rule}] {self.detail}"


@dataclass
class GuardReport:
    ok: bool
    syntax_error: dict | None = None
    violations: list[Violation] = field(default_factory=list)
    warnings: list[Violation] = field(default_factory=list)
    imported_modules: set[str] = field(default_factory=set)


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[Violation] = []
        self.warnings: list[Violation] = []
        self.aliases: dict[str, str] = {}  # local name -> root module ("o" -> "os")
        self.modules: set[str] = set()

    def _bad(self, node: ast.AST, rule: str, detail: str) -> None:
        self.violations.append(Violation(getattr(node, "lineno", 0), rule, detail))

    # ---- imports -------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            root = a.name.split(".")[0]
            self.modules.add(a.name)
            if root not in ALLOWED_MODULES:
                self._bad(node, "blocked-import", f"module '{a.name}' is not on the allowlist")
            self.aliases[(a.asname or a.name).split(".")[0]] = root
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        root = mod.split(".")[0]
        if node.level:
            self._bad(node, "blocked-import", "relative imports are not allowed")
        elif root not in ALLOWED_MODULES:
            self._bad(node, "blocked-import", f"module '{mod}' is not on the allowlist")
        else:
            self.modules.add(mod)
        for a in node.names:
            if a.name == "*":
                self._bad(node, "star-import", f"'from {mod} import *' hides what is imported")
            elif a.name in BLOCKED_MODULE_ATTRS.get(root, set()):
                self._bad(node, "blocked-attr", f"'{mod}.{a.name}' is forbidden")
        self.generic_visit(node)

    # ---- calls / names / attributes -----------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name) and f.id in BLOCKED_CALLS:
            self._bad(node, "blocked-call", f"call to '{f.id}()' is forbidden")
        if isinstance(f, ast.Name) and f.id in {"getattr", "setattr", "delattr", "hasattr"} and len(node.args) >= 2:
            arg = node.args[1]
            if not isinstance(arg, ast.Constant):
                self.warnings.append(Violation(node.lineno, "dynamic-attr", f"{f.id}() with a non-constant name"))
            elif isinstance(arg.value, str) and (arg.value.startswith("__") or arg.value in BLOCKED_DUNDERS):
                self._bad(node, "blocked-dunder", f"{f.id}(..., '{arg.value}') is forbidden")
        if isinstance(f, ast.Attribute) and f.attr == "show":
            self.warnings.append(Violation(node.lineno, "plt-show", "plt.show() does nothing in the sandbox; use savefig"))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in {"__builtins__", "__loader__", "__spec__"}:
            self._bad(node, "blocked-dunder", f"access to '{node.id}' is forbidden")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BLOCKED_DUNDERS:
            self._bad(node, "blocked-dunder", f"attribute '{node.attr}' is forbidden (sandbox-escape pattern)")
        elif node.attr in BLOCKED_ANY_ATTRS:
            self._bad(node, "blocked-attr", f"'{node.attr}' is forbidden (pickle = arbitrary code execution)")
        elif isinstance(node.value, ast.Name):
            root = self.aliases.get(node.value.id)
            if root and node.attr in BLOCKED_MODULE_ATTRS.get(root, set()):
                self._bad(node, "blocked-attr", f"'{root}.{node.attr}' is forbidden")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            has_exit = any(isinstance(n, (ast.Break, ast.Return, ast.Raise)) for n in ast.walk(node))
            if not has_exit:
                self.warnings.append(Violation(node.lineno, "infinite-loop", "'while True' with no break/return/raise"))
        self.generic_visit(node)


def check_code(code: str) -> GuardReport:
    try:
        tree = ast.parse(code, filename="solution.py")
    except SyntaxError as e:
        return GuardReport(ok=False, syntax_error={
            "type": type(e).__name__, "message": e.msg, "line": e.lineno or 0,
            "offset": e.offset or 0, "text": (e.text or "").rstrip("\n")})
    v = _Visitor()
    v.visit(tree)
    return GuardReport(ok=not v.violations, violations=v.violations, warnings=v.warnings,
                       imported_modules=v.modules)
