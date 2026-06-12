"""Evaluate the bad/good calibration archives and print a compact summary."""
import json

import config
import evaluator

for tag, gains in (("bad", config.CORRUPTED_GAINS),
                   ("good", config.KNOWN_GOOD_GAINS)):
    d = config.RESULTS_DIR / "baseline" / tag
    rep = evaluator.evaluate_suite(d, gains, 0)
    (d / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"=== {tag}: n_pass={rep['n_pass']}/8 overall_pass={rep['overall_pass']}")
    for tn, e in rep["tests"].items():
        m = json.dumps(e["metrics"], default=str)
        print(f"  {tn:18s} {e['status']:5s} {m[:160]}")
        if e["notes"]:
            print(f"  {'':18s}       note: {e['notes'][:140]}")
