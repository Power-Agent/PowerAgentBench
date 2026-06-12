"""Read PSS/e .out channel files via dyntools.

`import psse36` must precede `import dyntools` (it extends sys.path to the
PSS/e installation); both are imported lazily and cached so that modules which
never touch .out files don't need PSS/e at all.
"""
import json
from pathlib import Path

_dyntools = None


def _get_dyntools():
    global _dyntools
    if _dyntools is None:
        import psse36  # noqa: F401  (sets up sys.path for dyntools)
        import dyntools
        _dyntools = dyntools
    return _dyntools


def load_channels(out_path: Path):
    """Return (time: list[float], channels: dict[str, list[float]]).

    Channel keys are "<index>:<channel id string>".
    """
    dyntools = _get_dyntools()
    chnf = dyntools.CHNF(str(out_path))
    short_title, chanid, chandata = chnf.get_data()
    time = list(chandata["time"])
    channels = {}
    for idx, name in chanid.items():
        if idx == "time":
            continue
        channels[f"{idx}:{name}"] = list(chandata[idx])
    return time, channels


def dump_channel_map(results_dir: Path, dest_json: Path) -> dict:
    """Catalog every .out file in `results_dir`: channels, t_end, n_points."""
    catalog = {}
    for out_file in sorted(Path(results_dir).rglob("*.out")):
        try:
            time, channels = load_channels(out_file)
            catalog[out_file.name] = {
                "t_start": time[0] if time else None,
                "t_end": time[-1] if time else None,
                "n_points": len(time),
                "channels": sorted(channels.keys()),
            }
        except Exception as e:  # noqa: BLE001 - cataloging must not abort
            catalog[out_file.name] = {"error": f"{type(e).__name__}: {e}"}
    dest_json.parent.mkdir(parents=True, exist_ok=True)
    dest_json.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog
