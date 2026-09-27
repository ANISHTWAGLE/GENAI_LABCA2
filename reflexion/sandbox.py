"""Local execution sandbox (parent side).

Defence in depth - each layer catches what the previous one cannot:
  L1  static AST guard            (guard.py)        - before execution, precise line numbers
  L2  separate OS process         (this file)       - a crash / hang cannot take down the loop
  L3  runtime audit hook          (sandbox_boot.py) - blocks subprocess, network, writes outside the sandbox
  L4  OS resource limits          (this file)       - Windows Job Object: memory cap + max 1 process,
                                                      kill-on-close; POSIX: setrlimit
  L5  supervision                 (this file)       - wall-clock timeout, stdout/stderr size cap
  L6  hygiene                     (this file)       - fresh working dir per attempt, stripped env
                                                      (no API keys), stdin closed after handshake
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .guard import GuardReport, check_code

BOOT = Path(__file__).with_name("sandbox_boot.py")
MARKER = "[[SANDBOX-VIOLATION]]"
IS_WINDOWS = os.name == "nt"


@dataclass
class SandboxLimits:
    timeout_s: float = 30.0
    memory_mb: int = 512
    max_output_bytes: int = 2_000_000   # kill if stdout+stderr grow beyond this
    keep_output_chars: int = 12_000     # how much of each stream we keep for display/feedback


@dataclass
class ExecutionResult:
    status: str                    # OK | ERROR | TIMEOUT | MEMORY_LIMIT | OUTPUT_LIMIT | BLOCKED | SYNTAX | SANDBOX_FAILURE
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    workdir: Path
    peak_memory_mb: float | None = None
    guard: GuardReport | None = None
    error: dict | None = None      # structured record from _error.json (if the script raised)
    violations: list[dict] = field(default_factory=list)  # runtime audit-hook blocks

    @property
    def ok(self) -> bool:
        return self.status == "OK"


# --------------------------------------------------------------------------- Windows Job Object
if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BASIC_LIMITS(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _EXTENDED_LIMITS(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC_LIMITS), ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _EXT_INFO_CLASS = 9
    _LIMIT_ACTIVE_PROCESS = 0x0008
    _LIMIT_PROCESS_MEMORY = 0x0100
    _LIMIT_JOB_MEMORY = 0x0200
    _LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x0400
    _LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                               wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]


class _JobObject:
    """Windows kernel-enforced limits for the child (and anything it could spawn)."""

    def __init__(self, memory_mb: int, max_processes: int) -> None:
        self.handle = _k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = _EXTENDED_LIMITS()
        info.BasicLimitInformation.LimitFlags = (_LIMIT_PROCESS_MEMORY | _LIMIT_JOB_MEMORY | _LIMIT_ACTIVE_PROCESS
                                                 | _LIMIT_KILL_ON_JOB_CLOSE | _LIMIT_DIE_ON_UNHANDLED_EXCEPTION)
        info.BasicLimitInformation.ActiveProcessLimit = max_processes
        info.ProcessMemoryLimit = info.JobMemoryLimit = memory_mb * 1024 * 1024
        if not _k32.SetInformationJobObject(self.handle, _EXT_INFO_CLASS, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

    def assign(self, proc: subprocess.Popen) -> None:
        if not _k32.AssignProcessToJobObject(self.handle, int(proc._handle)):  # noqa: SLF001
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def peak_memory_mb(self) -> float | None:
        info = _EXTENDED_LIMITS()
        if _k32.QueryInformationJobObject(self.handle, _EXT_INFO_CLASS, ctypes.byref(info),
                                          ctypes.sizeof(info), None):
            return info.PeakJobMemoryUsed / (1024 * 1024)
        return None

    def kill(self) -> None:
        _k32.TerminateJobObject(self.handle, 1)

    def close(self) -> None:
        if self.handle:
            _k32.CloseHandle(self.handle)  # KILL_ON_JOB_CLOSE: anything still alive dies here
            self.handle = None


# --------------------------------------------------------------------------- runner
class Sandbox:
    def __init__(self, limits: SandboxLimits | None = None, input_files: dict[str, Path] | None = None,
                 cache_dir: Path | None = None) -> None:
        self.limits = limits or SandboxLimits()
        self.input_files = input_files or {}     # relative path inside sandbox -> source file (copied)
        self.cache_dir = cache_dir or Path(__file__).resolve().parent.parent / ".sandbox_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _prepare(self, workdir: Path, code: str) -> None:
        if workdir.exists():
            shutil.rmtree(workdir)
        (workdir / "outputs").mkdir(parents=True)
        (workdir / "tmp").mkdir()
        for rel, src in self.input_files.items():  # copies, never links: the script may write to them
            dst = workdir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        (workdir / "_sandbox.json").write_text(json.dumps({"write_roots": [str(workdir)]}))

    def _env(self, workdir: Path) -> dict[str, str]:
        """Minimal environment: nothing inherited except what Python needs to start."""
        env = {
            "PATH": os.pathsep.join([str(Path(sys.executable).parent)] +
                                    ([os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32")]
                                     if IS_WINDOWS else ["/usr/bin", "/bin"])),
            "TEMP": str(workdir / "tmp"), "TMP": str(workdir / "tmp"), "TMPDIR": str(workdir / "tmp"),
            "HOME": str(workdir), "USERPROFILE": str(workdir),
            "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(self.cache_dir / "matplotlib"),
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "LOKY_MAX_CPU_COUNT": "2",
        }
        if IS_WINDOWS:
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        return env

    def run(self, code: str, workdir: Path, *, skip_static: bool = False) -> ExecutionResult:
        workdir = Path(workdir).resolve()
        # L1: static guard - nothing runs if it fails
        report = check_code(code)
        if not skip_static and not report.ok:
            workdir.mkdir(parents=True, exist_ok=True)
            (workdir / "solution.py").write_text(code, encoding="utf-8")
            status = "SYNTAX" if report.syntax_error else "BLOCKED"
            return ExecutionResult(status, None, "", "", 0.0, workdir, guard=report,
                                   error=report.syntax_error)
        self._prepare(workdir, code)
        return self._execute(workdir, report)

    def _execute(self, workdir: Path, report: GuardReport) -> ExecutionResult:
        lim = self.limits
        out_path, err_path = workdir / "_stdout.txt", workdir / "_stderr.txt"
        cmd = [sys.executable, "-X", "utf8", "-B", "-s", str(BOOT), "solution.py", "_sandbox.json"]
        kwargs: dict = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = _posix_limits(lim)

        status, job, peak = "OK", None, None
        start = time.monotonic()
        with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
            try:
                proc = subprocess.Popen(cmd, cwd=workdir, env=self._env(workdir), stdin=subprocess.PIPE,
                                        stdout=out_f, stderr=err_f, **kwargs)
            except OSError as e:
                return ExecutionResult("SANDBOX_FAILURE", None, "", f"could not start sandbox: {e}", 0.0,
                                       workdir, guard=report)
            try:
                if IS_WINDOWS:
                    # a venv python.exe is a launcher that spawns the real interpreter -> allow 2
                    job = _JobObject(lim.memory_mb, 1 if sys.prefix == sys.base_prefix else 2)
                    job.assign(proc)
                proc.stdin.write(b"GO\n")   # limits are in place: let the child proceed
                proc.stdin.close()          # the script gets EOF on any input() attempt
                while proc.poll() is None:
                    elapsed = time.monotonic() - start
                    if elapsed > lim.timeout_s:
                        status = "TIMEOUT"
                        break
                    if out_path.stat().st_size + err_path.stat().st_size > lim.max_output_bytes:
                        status = "OUTPUT_LIMIT"
                        break
                    time.sleep(0.05)
                if status != "OK":
                    _kill(proc, job)
                proc.wait(timeout=10)
            finally:
                if job is not None:
                    peak = job.peak_memory_mb()
                    job.close()
                elif proc.poll() is None:
                    _kill(proc, None)
        duration = time.monotonic() - start

        stdout = _read_capped(out_path, lim.keep_output_chars)
        stderr_full = _read_capped(err_path, 200_000)
        violations = _extract_violations(stderr_full)
        stderr = _strip_markers(stderr_full)[-lim.keep_output_chars:]
        error = None
        err_json = workdir / "_error.json"
        if err_json.exists():
            try:
                error = json.loads(err_json.read_text(encoding="utf-8")[:50_000])
            except (json.JSONDecodeError, OSError):
                error = None

        exit_code = proc.returncode
        if status == "OK":
            if error and error.get("type") == "MemoryError" or _looks_oom(exit_code, stderr, peak, lim):
                status = "MEMORY_LIMIT"
            elif violations:
                status = "BLOCKED"   # even if the script swallowed the exception and exited 0
            elif exit_code != 0:
                status = "ERROR"
        return ExecutionResult(status, exit_code, stdout, stderr, round(duration, 2), workdir,
                               peak_memory_mb=round(peak, 1) if peak else None, guard=report,
                               error=error, violations=violations)


def _kill(proc: subprocess.Popen, job: _JobObject | None) -> None:
    if job is not None:
        job.kill()
    elif not IS_WINDOWS:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        proc.kill()


def _posix_limits(lim: SandboxLimits):
    def apply() -> None:
        import resource
        mem = lim.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        cpu = int(lim.timeout_s) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024,) * 2)
    return apply


def _looks_oom(exit_code, stderr: str, peak, lim: SandboxLimits) -> bool:
    if "MemoryError" in stderr[-3000:]:
        return True
    # Windows: allocation failure inside C code can terminate the process abruptly
    return bool(exit_code not in (0, 1) and peak and peak >= lim.memory_mb * 0.95)


def _read_capped(path: Path, n: int) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > n:
                head = fh.read(n // 4).decode("utf-8", "replace")
                fh.seek(size - (3 * n) // 4)
                tail = fh.read().decode("utf-8", "replace")
                return f"{head}\n... [{size - n:,} bytes truncated] ...\n{tail}"
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _extract_violations(stderr: str) -> list[dict]:
    out = []
    for line in stderr.splitlines():
        if line.startswith(MARKER):
            try:
                out.append(json.loads(line[len(MARKER):].strip()))
            except json.JSONDecodeError:
                out.append({"event": "unknown", "detail": line[len(MARKER):].strip()[:200]})
    return out


def _strip_markers(stderr: str) -> str:
    return "\n".join(l for l in stderr.splitlines() if not l.startswith(MARKER))
