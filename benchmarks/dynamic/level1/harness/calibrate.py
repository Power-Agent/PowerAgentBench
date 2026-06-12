"""Calibration / verification runs (gates before any agent run).

Usage (with the PSS/e-bound Python 3.11):
    python calibrate.py bad      # corrupted gains (10,50,10,50) baseline
    python calibrate.py good     # known-good gains (1,5,1,5) from the paper
    python calibrate.py discover # dump channel_map.json from the 'bad' archive

Each run leaves:
    AgentBench\\results\\baseline\\work_<tag>\\   (case working copy)
    AgentBench\\results\\baseline\\<tag>\\        (archived RESULTs + stdout)
"""
import shutil
import sys
from pathlib import Path

import channels
import config
import dmview_runner
import dyr_tools


def _setup_workdir(tag: str, gains: dict | None) -> Path:
    workdir = config.RESULTS_DIR / "baseline" / f"work_{tag}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    for fn in ("Solar.sav", "Solar.dyr"):
        shutil.copy2(config.PRISTINE_DIR / fn, workdir / fn)
    if gains is not None:
        dyr_tools.write_gains(workdir / "Solar.dyr", gains)
    print(f"[{tag}] gains = {dyr_tools.read_gains(workdir / 'Solar.dyr')}")
    return workdir


def run_case(tag: str, gains: dict | None) -> None:
    workdir = _setup_workdir(tag, gains)
    archive = config.RESULTS_DIR / "baseline" / tag
    if archive.exists():
        shutil.rmtree(archive)
    print(f"[{tag}] running DMView suite ...")
    res = dmview_runner.run_suite(workdir, archive)
    print(f"[{tag}] suite status={res.status} elapsed={res.elapsed_s:.0f}s "
          f"rc={res.returncode}")
    print(f"[{tag}] archived to {res.archive_dir}")
    out_files = sorted((archive / "RESULTs").glob("*"))
    print(f"[{tag}] artifacts: {[p.name for p in out_files]}")
    cmap = channels.dump_channel_map(archive / "RESULTs",
                                     archive / "channel_map.json")
    n_ok = sum(1 for v in cmap.values() if "error" not in v)
    print(f"[{tag}] channel map: {n_ok}/{len(cmap)} .out files readable")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bad"
    if cmd == "bad":
        run_case("bad", None)  # pristine file already holds the corrupted gains
    elif cmd == "good":
        run_case("good", config.KNOWN_GOOD_GAINS)
    elif cmd == "discover":
        archive = config.RESULTS_DIR / "baseline" / "bad"
        channels.dump_channel_map(archive / "RESULTs",
                                  archive / "channel_map.json")
        print("channel map written")
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
