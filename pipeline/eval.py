"""Score pipeline predictions against the team labels (primary) and the organizer labels.

Predictions: a dir of per-clip result JSON files (`X-NN.json`, the Analyzer result schema) or one JSON file
{clip: result}. Ground truth: analysis/gt_team.json ({clip: {verdict, t, types, org_verdict, duplicate_of}}).

Per clip:
- verdict match (team / organizer);
- type hit: any predicted type_id is among the team types;
- timecode error: |t_pred - t_label| for the predicted violation closest to the label (team t, else organizer t).

Usage: python -m pipeline.eval PRED_DIR_OR_JSON [--gt analysis/gt_team.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib

DEFAULT_GT = pathlib.Path(__file__).resolve().parent.parent / "analysis/gt_team.json"


def load_preds(path: pathlib.Path) -> dict:
    if path.is_dir():
        return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("X-*.json"))}
    return json.loads(path.read_text(encoding="utf-8"))


def score(preds: dict, gt: dict) -> dict:
    rows = []
    for clip, g in sorted(gt.items()):
        if clip not in preds:
            continue
        p = preds[clip]
        viol = p.get("violations") or []
        pv = "violation" if viol else "no_violation"
        types = [v.get("type_id") for v in viol]
        t_label = g.get("t")
        terr = None
        if viol and t_label is not None:
            terr = min(abs(v["timecode_sec"] - t_label) for v in viol)
        rows.append({
            "clip": clip, "pred": pv, "team": g["verdict"], "org": g["org_verdict"],
            "ok_team": pv == g["verdict"], "ok_org": pv == g["org_verdict"],
            "types_pred": types, "types_gt": g.get("types") or [],
            "type_hit": bool(set(types) & set(g.get("types") or [])) if g["verdict"] == "violation" else None,
            "t_label": t_label, "t_pred": [v["timecode_sec"] for v in viol], "t_err": terr,
        })
    n = len(rows)
    viol_rows = [r for r in rows if r["team"] == "violation"]
    clean_rows = [r for r in rows if r["team"] == "no_violation"]
    errs = [r["t_err"] for r in rows if r["t_err"] is not None and r["team"] == "violation"]
    summary = {
        "clips": n,
        "verdict_acc_team": sum(r["ok_team"] for r in rows) / max(n, 1),
        "verdict_acc_org": sum(r["ok_org"] for r in rows) / max(n, 1),
        "recall_team": sum(r["pred"] == "violation" for r in viol_rows) / max(len(viol_rows), 1),
        "false_alarms_team": sum(r["pred"] == "violation" for r in clean_rows),
        "clean_clips": len(clean_rows),
        "type_hit": sum(bool(r["type_hit"]) for r in viol_rows) / max(len(viol_rows), 1),
        "t_err_le2": sum(e <= 2 for e in errs) / max(len(errs), 1) if errs else None,
        "t_err_median": sorted(errs)[len(errs) // 2] if errs else None,
    }
    return {"summary": summary, "rows": rows}


def print_report(res: dict) -> None:
    for r in res["rows"]:
        mark = "OK " if r["ok_team"] else "ERR"
        tp = ",".join(f"{t:g}" for t in r["t_pred"])
        print(f"{mark} {r['clip']} pred={r['pred'][:5]:5} team={r['team'][:5]:5} org={r['org'][:5]:5} "
              f"types={'/'.join(map(str, r['types_pred'])) or '-':14} gt={'/'.join(r['types_gt']) or '-':10} "
              f"t={tp or '-':12} label={r['t_label']} err={r['t_err']}")
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--gt", default=str(DEFAULT_GT))
    a = ap.parse_args()
    gt = json.loads(pathlib.Path(a.gt).read_text(encoding="utf-8"))
    print_report(score(load_preds(pathlib.Path(a.pred)), gt))


if __name__ == "__main__":
    main()
