"""Child-process bootstrap: arms the runtime guard, then runs the untrusted script.

Executed as:  python -X utf8 -B sandbox_boot.py <solution.py> <config.json>
with cwd = the per-attempt sandbox directory. Standard library only.

Order matters:
  1. wait for the parent's GO line (parent first puts us in the Job Object -> no race)
  2. pre-import the *trusted* scientific libraries the script asks for (before the hook,
     so their one-off initialisation - font caches, DLL loading - is unaffected)
  3. install a PEP 578 audit hook (cannot be removed once installed)
  4. run the script; on failure, emit a structured error record (_error.json)

Every blocked operation is also written straight to file descriptor 2 with a marker,
so the parent sees it even if the script catches the exception and carries on.
"""
import ast
import json
import os
import sys
import traceback

MARKER = "[[SANDBOX-VIOLATION]]"
TRUSTED_PREIMPORT = ("numpy", "pandas", "matplotlib", "scipy", "sklearn")

BLOCKED_EVENTS = {
    "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty",
    "os.kill", "os.killpg", "os.startfile", "pty.spawn", "os.putenv", "os.unsetenv",
    "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto",
    "socket.sendmsg", "urllib.Request", "ftplib.connect", "smtplib.connect", "webbrowser.open",
    "sys.addaudithook", "sys.settrace", "sys.setprofile",
}
PATH_EVENTS = {  # event -> indices of path arguments that must stay inside the sandbox
    "os.remove": (0,), "os.rmdir": (0,), "os.rename": (0, 1), "os.replace": (0, 1), "os.mkdir": (0,),
    "os.chmod": (0,), "os.truncate": (0,), "shutil.rmtree": (0,), "shutil.move": (0, 1),
    "shutil.copyfile": (1,), "os.link": (0, 1), "os.symlink": (0, 1), "os.utime": (0,),
}
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


class SandboxViolation(PermissionError):
    """Raised by the audit hook when the script attempts a forbidden operation."""


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def main() -> None:
    solution, config_path = sys.argv[1], sys.argv[2]
    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    roots = [_norm(r) for r in cfg["write_roots"]]
    solution = os.path.abspath(solution)
    sys.path[:] = [p for p in sys.path if p and _norm(p) != _norm(os.path.dirname(__file__))]
    sys.path.insert(0, os.getcwd())

    # 1. handshake: parent assigns resource limits, then says GO
    if sys.stdin.readline().strip() != "GO":
        sys.exit(97)

    # 2. pre-import trusted libraries requested by the script
    with open(solution, encoding="utf-8") as fh:
        source = fh.read()
    try:
        for node in ast.walk(ast.parse(source)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
            for name in names:
                if name.split(".")[0] in TRUSTED_PREIMPORT:
                    try:
                        __import__(name)
                    except Exception:
                        pass  # the script's own import will raise with a proper traceback
    except SyntaxError:
        pass

    # 3. arm the audit hook
    devnull = _norm(os.devnull)
    state = {"busy": False}

    def inside(path) -> bool:
        if isinstance(path, int):
            return True
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        p = _norm(os.fspath(path))
        return p == devnull or any(p == r or p.startswith(r + os.sep) for r in roots)

    def deny(event: str, detail: str):
        msg = f"{event} blocked by sandbox ({detail})"
        try:
            os.write(2, f"\n{MARKER} {json.dumps({'event': event, 'detail': detail})}\n".encode())
        except OSError:
            pass
        raise SandboxViolation(msg)

    def hook(event: str, args: tuple) -> None:
        if state["busy"]:
            return
        state["busy"] = True
        try:
            if event == "open":
                path, mode, flags = (tuple(args) + (None, None, None))[:3]
                writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                          (isinstance(flags, int) and flags & WRITE_FLAGS)
                if writing and not inside(path):
                    deny(event, f"write outside sandbox: {path!r}")
            elif event in BLOCKED_EVENTS:
                deny(event, repr(args)[:160])
            elif event in PATH_EVENTS:
                for i in PATH_EVENTS[event]:
                    if i < len(args) and args[i] is not None and not inside(args[i]):
                        deny(event, f"path outside sandbox: {args[i]!r}")
        finally:
            state["busy"] = False

    sys.addaudithook(hook)

    # 4. run the untrusted script as __main__
    import runpy
    try:
        runpy.run_path(solution, run_name="__main__")
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - we report everything
        _report(exc, solution)
        sys.exit(1)


def _report(exc: BaseException, solution: str) -> None:
    """Print a traceback trimmed to the script's frames and write a structured record."""
    tb = exc.__traceback__
    while tb is not None and _norm(tb.tb_frame.f_code.co_filename) != _norm(solution):
        tb = tb.tb_next
    all_frames = traceback.extract_tb(exc.__traceback__)
    user = [f for f in all_frames if _norm(f.filename) == _norm(solution)]
    lib = [f for f in all_frames if _norm(f.filename) != _norm(solution)]
    try:
        traceback.print_exception(type(exc), exc, tb if tb is not None else exc.__traceback__)
    except Exception:
        pass
    cause = exc.__cause__ or exc.__context__
    record = {
        "type": type(exc).__name__,
        "message": str(exc)[:2000],
        "user_frames": [{"line": f.lineno, "func": f.name, "code": f.line} for f in user],
        "last_library_frame": ({"file": os.path.basename(lib[-1].filename), "line": lib[-1].lineno,
                                "func": lib[-1].name} if lib and all_frames[-1] is lib[-1] else None),
        "cause": f"{type(cause).__name__}: {str(cause)[:300]}" if cause else None,
    }
    try:
        with open("_error.json", "w", encoding="utf-8") as fh:
            json.dump(record, fh)
    except Exception:
        pass


if __name__ == "__main__":
    main()
