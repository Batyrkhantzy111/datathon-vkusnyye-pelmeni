"""Score augmented-clip results (X-NN__aug.json) against the original clip's team label and summarize signals.

Usage (GPU host): ~/det-venv/bin/python analysis/robust/score.py ~/forte/cache/robust [--orig ~/forte/cache/results]
Prints one line per clip x aug (verdict/type/timecode vs label + layout, crowd, glass-line diagnostics), per-aug
summaries (pipeline.eval.score) and a JSON dump to stdout's last line.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from pipeline import eval as ev  # noqa: E402


def diag(sig: dict | None) -> dict:
    if not sig:
        return {}
    p = sig.get("persons") or {}
    fr = p.get("frames") or []
    cls = {}
    feet_above = 0
    for f in fr:
        for b in f["boxes"]:
            cls[b[6]] = cls.get(b[6], 0) + 1
    n_room = [f["n_room"] for f in fr]
    roles = {}
    for t in p.get("tracks") or []:
        roles[t["role"]] = roles.get(t["role"], 0) + 1
    st = sig.get("states") or []
    ppl = [s.get("people_in_room") for s in st if s.get("people_in_room") is not None]
    return {
        "layout": sorted((sig.get("layout") or {}).keys()),
        "glass_y": (p.get("meta") or {}).get("glass_line_y"),
        "box_cls": cls,
        "max_room": max(n_room) if n_room else None,
        "mean_room": round(sum(n_room) / len(n_room), 2) if n_room else None,
        "roles": roles,
        "crowd": p.get("crowd_intervals"),
        "crowd_checks": len(p.get("crowd_checks") or []),
        "events": [f"{e['type']}@{e['t']}" for e in p.get("events") or []],
        "vlm_ppl_max": max(ppl) if ppl else None,
        "states_ok": f"{sum(bool(s.get('ok')) for s in st)}/{len(st)}",
        "cands": [f"{c['type_id']}@{round(c['t'], 1)}" for c in sig.get("candidates") or []],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--orig", default=str(pathlib.Path.home() / "forte/cache/results"))
    ap.add_argument("--gt", default=str(pathlib.Path.home() / "forte/analysis/gt_team.json"))
    a = ap.parse_args()
    gt = json.loads(pathlib.Path(a.gt).read_text(encoding="utf-8"))
    by_aug: dict[str, dict] = {}
    sigs: dict[str, dict] = {}
    runtime: dict[str, dict] = {}
    for p in sorted(pathlib.Path(a.pred).glob("X-*__*.json")):
        if p.name.endswith(".signals.json"):
            continue
        clip, aug = p.stem.split("__")
        res = json.loads(p.read_text(encoding="utf-8"))
        by_aug.setdefault(aug, {})[clip] = res
        runtime[p.stem] = {"sec": res["meta"]["processing_sec"], "dur": res["meta"]["duration_sec"],
                           "stages": res["meta"]["stage_sec"]}
        s = p.with_name(p.stem + ".signals.json")
        sigs[p.stem] = diag(json.loads(s.read_text(encoding="utf-8")) if s.exists() else None)
    orig = pathlib.Path(a.orig)
    if orig.exists():
        clips = sorted({c for d in by_aug.values() for c in d})
        o = {c: json.loads((orig / f"{c}.json").read_text(encoding="utf-8")) for c in clips
             if (orig / f"{c}.json").exists()}
        by_aug = {"orig": o, **by_aug}
        for c in o:
            s = orig / f"{c}.signals.json"
            sigs[f"{c}__orig"] = diag(json.loads(s.read_text(encoding="utf-8")) if s.exists() else None)
    out = {"summary": {}, "rows": [], "runtime": runtime}
    for aug, preds in by_aug.items():
        sc = ev.score(preds, gt)
        out["summary"][aug] = sc["summary"]
        print(f"===== {aug}")
        for r in sc["rows"]:
            key = f"{r['clip']}__{aug}"
            r["aug"] = aug
            r["diag"] = sigs.get(key, {})
            out["rows"].append(r)
            d = r["diag"]
            mark = "OK " if r["ok_team"] else "ERR"
            print(f"{mark} {key:16} types={'/'.join(map(str, r['types_pred'])) or '-':10} gt={'/'.join(r['types_gt']) or '-':4}"
                  f" t={r['t_pred']} label={r['t_label']} err={r['t_err']} | lay={','.join(d.get('layout', []))}"
                  f" cls={d.get('box_cls')} maxroom={d.get('max_room')} vlm_ppl={d.get('vlm_ppl_max')}"
                  f" crowd={d.get('crowd')} cc={d.get('crowd_checks')} roles={d.get('roles')}"
                  f" ev={d.get('events')} rt={runtime.get(key, {}).get('sec')}")
        print(json.dumps(sc["summary"], ensure_ascii=False))
    print("JSON " + json.dumps(out, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
