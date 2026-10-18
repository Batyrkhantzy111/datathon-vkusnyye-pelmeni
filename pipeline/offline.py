"""Offline rule tuning: run rules.evaluate on cached signals of the open clips and score against the labels.

Caches (GPU host): ~/forte/cache/{states,persons,checks,facts}/X-NN.json. Missing caches are treated as absent
signals. Prints the per-clip table and the summary from pipeline.eval.

Usage: ~/det-venv/bin/python -m pipeline.offline [--cache ~/forte/cache] [--gt ~/forte/analysis/gt_team.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib

from pipeline import eval as ev
from pipeline import rules

NORMS = json.loads((pathlib.Path(__file__).with_name("norms.json")).read_text(encoding="utf-8"))


def _load(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def signals_for(cache: pathlib.Path, clip: str) -> rules.Signals | None:
    st = _load(cache / "states" / f"{clip}.json")
    if st is None:
        return None
    persons = _load(cache / "persons" / f"{clip}.json")
    checks: dict = {"leave": [], "entry": []}
    for f in sorted((cache / "checks").glob(f"{clip}*.json")):
        d = _load(f)
        for rec in d if isinstance(d, list) else [d]:
            if not isinstance(rec, dict) or "result" not in rec:
                continue
            r = dict(rec["result"])
            if rec.get("kind") == "entry":
                r["t_enter"] = rec.get("t_enter")
                checks["entry"].append(r)
            elif rec.get("kind") == "leave":
                r["t_leave"] = rec.get("t_leave")
                r["t_stand"] = rec.get("t_stand")
                checks["leave"].append(r)
    facts = _load(cache / "facts" / f"{clip}.json")
    if isinstance(facts, dict) and "windows" in facts:
        facts = facts["windows"]
    return rules.Signals(duration=st.get("duration", 0.0), states=st["states"], persons=persons, checks=checks,
                         facts=facts)


def predict(cache: pathlib.Path, clips: list[str], params: dict | None = None) -> dict:
    preds = {}
    for clip in clips:
        sig = signals_for(cache, clip)
        if sig is None:
            continue
        cands = rules.evaluate(sig, params)
        preds[clip] = {"violations": [{"type_id": c["type_id"], "timecode_sec": c["t"],
                                       "type": NORMS["types"].get(c["type_id"], {}).get("name"),
                                       "observation": c["observation"]} for c in cands]}
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(pathlib.Path.home() / "forte/cache"))
    ap.add_argument("--gt", default=str(pathlib.Path.home() / "forte/analysis/gt_team.json"))
    ap.add_argument("--show", action="store_true", help="print observations of every prediction")
    a = ap.parse_args()
    gt = json.loads(pathlib.Path(a.gt).read_text(encoding="utf-8"))
    preds = predict(pathlib.Path(a.cache), sorted(gt))
    res = ev.score(preds, gt)
    ev.print_report(res)
    if a.show:
        for clip, p in preds.items():
            for v in p["violations"]:
                print(f"{clip} {v['type_id']} {v['timecode_sec']}: {v['observation']}")


if __name__ == "__main__":
    main()
