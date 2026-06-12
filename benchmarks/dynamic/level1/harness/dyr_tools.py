"""Constrained editor for the REECAU1 gains in a PSS/e .dyr file.

The benchmark action space is exactly the four electrical-control gains
Kqp, Kqi, Kvp, Kvi, which sit at data-token positions 24-27 (1-indexed,
counting ICONs then CONs) of the REECAU1 USRMDL record:

    91003, 'USRMDL', S1, 'REECAU1', IC, IT, NI, NC, NS, NV,
    <NI icon tokens> <NC con tokens> /

Everything else in the file is locked: writes splice replacement text into
the exact character spans of those four tokens, leaving every other byte
untouched.
"""
import json
import math
import re
import time
from pathlib import Path

import config

_HEADER_TOKENS = 10           # bus, 'USRMDL', mc id, 'REECAU1', IC, IT, NI, NC, NS, NV
_GAIN_POSITIONS = (24, 25, 26, 27)   # 1-indexed within the data tokens


class DyrEditError(Exception):
    """Raised when a read/write violates the locked-file constraints."""


def _locate_record(text: str):
    """Return (tokens, spans) for the unique REECAU1 record.

    tokens: list of token strings for the whole record (header + data).
    spans:  matching (start, end) character offsets into `text`.
    """
    starts = [m.start() for m in re.finditer(r"(?im)^.*'REECAU1'.*$", text)]
    if len(starts) != 1:
        raise DyrEditError(f"expected exactly one REECAU1 record, found {len(starts)}")
    rec_start = starts[0]
    term = text.find("/", rec_start)
    if term == -1:
        raise DyrEditError("REECAU1 record has no '/' terminator")

    tokens, spans = [], []
    for m in re.finditer(r"[^\s,]+", text[rec_start:term]):
        tokens.append(m.group(0))
        spans.append((rec_start + m.start(), rec_start + m.end()))

    if len(tokens) < _HEADER_TOKENS:
        raise DyrEditError("REECAU1 record too short to contain a USRMDL header")
    if "USRMDL" not in tokens[1].upper() or "REECAU1" not in tokens[3].upper():
        raise DyrEditError(f"unexpected REECAU1 header layout: {tokens[:_HEADER_TOKENS]}")

    ni, nc = int(tokens[6]), int(tokens[7])
    n_data = len(tokens) - _HEADER_TOKENS
    if n_data != ni + nc:
        raise DyrEditError(
            f"REECAU1 data token count {n_data} != NI+NC = {ni + nc}; file structure changed"
        )
    if n_data < max(_GAIN_POSITIONS):
        raise DyrEditError("REECAU1 record has fewer data tokens than the gain positions")
    return tokens, spans


def _gain_indices(tokens):
    """Record-token indices of the four gains."""
    return [_HEADER_TOKENS + p - 1 for p in _GAIN_POSITIONS]


def read_gains(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    tokens, _ = _locate_record(text)
    out = {}
    for name, idx in zip(config.GAIN_NAMES, _gain_indices(tokens)):
        try:
            out[name] = float(tokens[idx])
        except ValueError as e:
            raise DyrEditError(f"gain token {name} is not numeric: {tokens[idx]!r}") from e
    return out


def validate_gains(gains: dict) -> str | None:
    """Return an error message, or None if the gains are acceptable."""
    for name in config.GAIN_NAMES:
        if name not in gains:
            return f"missing value for {name}"
        v = gains[name]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return f"{name} must be a number, got {type(v).__name__}"
        if not math.isfinite(v):
            return f"{name} must be finite, got {v}"
        if not (config.GAIN_MIN < v <= config.GAIN_MAX):
            return (f"{name}={v} out of allowed range "
                    f"({config.GAIN_MIN}, {config.GAIN_MAX}]")
    return None


def write_gains(path: Path, gains: dict, log_path: Path | None = None,
                iteration: int | None = None) -> dict:
    """Replace only the four gain tokens; returns {'old': {...}, 'new': {...}}."""
    err = validate_gains(gains)
    if err:
        raise DyrEditError(err)

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    tokens, spans = _locate_record(text)
    idxs = _gain_indices(tokens)

    old = {n: float(tokens[i]) for n, i in zip(config.GAIN_NAMES, idxs)}
    new = {n: float(gains[n]) for n in config.GAIN_NAMES}

    for name, idx in sorted(zip(config.GAIN_NAMES, idxs),
                            key=lambda t: t[1], reverse=True):
        s, e = spans[idx]
        text = text[:s] + format(new[name], "g") + text[e:]

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

    readback = read_gains(path)
    if any(abs(readback[n] - new[n]) > 1e-12 for n in config.GAIN_NAMES):
        raise DyrEditError(f"post-write verification failed: {readback} != {new}")

    if log_path is not None:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "iteration": iteration, "old": old, "new": new}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    return {"old": old, "new": new}


def _self_test():
    """Round-trip test on a scratch copy of the pristine file."""
    import shutil
    import tempfile

    src = config.PRISTINE_DIR / "Solar.dyr"
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "Solar.dyr"
        shutil.copy2(src, work)

        g0 = read_gains(work)
        assert g0 == config.CORRUPTED_GAINS, g0

        res = write_gains(work, config.KNOWN_GOOD_GAINS, Path(td) / "log.jsonl", 1)
        assert res["old"] == config.CORRUPTED_GAINS and res["new"] == config.KNOWN_GOOD_GAINS

        # everything outside the 4 tokens must be byte-identical
        orig, edited = src.read_text(), work.read_text()
        t0, s0 = _locate_record(orig)
        t1, s1 = _locate_record(edited)
        idxs = set(_gain_indices(t0))
        assert len(t0) == len(t1)
        for i, (a, b) in enumerate(zip(t0, t1)):
            assert (a == b) or (i in idxs), f"token {i} changed unexpectedly: {a} -> {b}"
        pre = orig[: s0[min(idxs)][0]]
        assert edited.startswith(pre), "bytes before the gain tokens changed"
        post = orig[s0[max(idxs)][1]:]
        assert edited.endswith(post), "bytes after the gain tokens changed"

        # invalid inputs must be rejected
        for bad in [
            {"Kqp": 0, "Kqi": 5, "Kvp": 1, "Kvi": 5},      # 0 is out of (0, 100]
            {"Kqp": 101, "Kqi": 5, "Kvp": 1, "Kvi": 5},    # > 100
            {"Kqp": float("nan"), "Kqi": 5, "Kvp": 1, "Kvi": 5},
            {"Kqp": "2", "Kqi": 5, "Kvp": 1, "Kvi": 5},    # string
            {"Kqi": 5, "Kvp": 1, "Kvi": 5},                # missing key
        ]:
            assert validate_gains(bad) is not None, bad

    print("dyr_tools self-test: OK")


if __name__ == "__main__":
    _self_test()
