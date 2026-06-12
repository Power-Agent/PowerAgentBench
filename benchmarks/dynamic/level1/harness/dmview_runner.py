"""Execute one DMView test-suite run and archive its outputs.

DMView writes everything into the shared RESULTs\\ folder at the DMView root,
so this module owns that folder: it asserts single-flight via a lock file,
clears the folder before a run, and moves all outputs into the caller's
archive folder afterwards.
"""
import shutil
import subprocess
import time
from pathlib import Path

import config
import ini_gen

_LICENSE_MARKERS = ("license", "licence", "no license")


class SuiteResult:
    def __init__(self, status: str, elapsed_s: float, archive_dir: Path,
                 returncode: int | None, stdout_tail: str):
        self.status = status            # completed | dmview_error | timeout
        self.elapsed_s = elapsed_s
        self.archive_dir = archive_dir
        self.returncode = returncode
        self.stdout_tail = stdout_tail


def _clear_results_dir():
    config.RESULTS_GLOBAL.mkdir(exist_ok=True)
    for p in config.RESULTS_GLOBAL.iterdir():
        if p.name == config.RESULTS_LOCK.name:
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def _acquire_lock():
    config.RESULTS_GLOBAL.mkdir(exist_ok=True)
    if config.RESULTS_LOCK.exists():
        raise RuntimeError(
            f"{config.RESULTS_LOCK} exists - another suite appears to be running. "
            "Delete the lock file if that is stale."
        )
    config.RESULTS_LOCK.write_text(str(time.time()), encoding="utf-8")


def _release_lock():
    try:
        config.RESULTS_LOCK.unlink()
    except FileNotFoundError:
        pass


def _launch() -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen(
        [str(config.PY311), "DMView.py", config.INI_NAME],
        cwd=str(config.DMVIEW_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )
    try:
        out, _ = proc.communicate(timeout=config.SUITE_TIMEOUT_S)
        return proc, out or ""
    except subprocess.TimeoutExpired:
        # kill the whole tree: PSS/e can leave child processes behind
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
        try:
            out, _ = proc.communicate(timeout=30)
        except Exception:
            out = ""
        proc.returncode = proc.returncode if proc.returncode is not None else -1
        raise subprocess.TimeoutExpired(cmd="DMView", timeout=config.SUITE_TIMEOUT_S,
                                        output=out)


def _archive(dest: Path, stdout_text: str):
    dest.mkdir(parents=True, exist_ok=True)
    results = dest / "RESULTs"
    results.mkdir(exist_ok=True)
    for p in list(config.RESULTS_GLOBAL.iterdir()):
        if p.name == config.RESULTS_LOCK.name:
            continue
        shutil.move(str(p), str(results / p.name))
    (dest / "stdout.txt").write_text(stdout_text, encoding="utf-8")


def run_suite(workdir: Path, archive_dir: Path, retry_on_license: bool = True) -> SuiteResult:
    """Run the full DMView suite on `workdir`'s case; archive into `archive_dir`."""
    ini_gen.write_benchmark_ini(workdir)
    _acquire_lock()
    t0 = time.time()
    try:
        _clear_results_dir()
        try:
            proc, out = _launch()
            timed_out = False
        except subprocess.TimeoutExpired as e:
            out = e.output or ""
            proc = None
            timed_out = True

        elapsed = time.time() - t0
        _archive(archive_dir, out)

        n_out_files = len(list((archive_dir / "RESULTs").rglob("*.out")))
        low = out.lower()

        if timed_out:
            status = "timeout"
        elif n_out_files == 0:
            status = "dmview_error"
            if retry_on_license and any(m in low for m in _LICENSE_MARKERS):
                _release_lock()
                time.sleep(60)
                return run_suite(workdir, archive_dir, retry_on_license=False)
        else:
            # DMView's exit code is unreliable; presence of .out files is the signal
            status = "completed"

        return SuiteResult(
            status=status, elapsed_s=elapsed, archive_dir=archive_dir,
            returncode=(proc.returncode if proc else None),
            stdout_tail=out[-2000:],
        )
    finally:
        _release_lock()
