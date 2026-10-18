"""Run and score `pipeline.vlm_facts` on the open clips.

run:    decode each clip at 2 fps, ask the window facts, cache {config, runtime_sec, windows} to OUT/X-NN.json
        (clips run `--parallel` at a time against the shared vLLM server, so runtime_sec includes queueing)
report: per fact x clip table (clip-level: fact seen in >= votes windows) against the expected facts below,
        plus the derived visit features the rules engine needs (handover structure for ID-before-cash,
        return of notes, recording, machine count of a bag).

Expected facts come from the scene narratives in analysis/our_labels.json (`desc`), written by the team.
Clips not listed for a fact are not scored for it.

GPU host, from ~/forte:
  ~/det-venv/bin/python -m pipeline.eval_facts run --out ~/forte/cache/facts [--clips X-01,X-02] [--crop 0.1,0.7]
  ~/det-venv/bin/python -m pipeline.eval_facts report --out ~/forte/cache/facts [--votes 2]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

from pipeline.vlm import VlmClient
from pipeline.vlm_facts import FACT_NAMES, desk_roi, session_facts, video_facts

ALL = ["X-%02d" % i for i in (1, 2, 3, 4, 5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                              26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 43, 44, 45, 46, 47, 48)]

# fact -> (clips where the narrative shows it, clips where the narrative rules it out)
_SEATED_NO_CASH = ["X-11", "X-12", "X-22", "X-23", "X-24", "X-25", "X-28", "X-29", "X-30", "X-31", "X-32",
                   "X-33", "X-40"]
EXPECTED: dict[str, tuple[list[str], list[str]]] = {
    "id_document": (["X-01", "X-03", "X-04", "X-05", "X-16", "X-18", "X-19", "X-47", "X-48"],
                    ["X-02", "X-17", "X-34", "X-35", "X-37", "X-08", "X-10", "X-13", "X-14", "X-43", "X-44",
                     "X-45", "X-46", *_SEATED_NO_CASH]),
    "cash_received": (["X-01", "X-02", "X-03", "X-04", "X-05", "X-16", "X-17", "X-18", "X-19", "X-34", "X-35",
                       "X-37", "X-38", "X-39", "X-47", "X-48"],
                      ["X-08", "X-10", "X-13", "X-14", "X-20", "X-21", "X-43", "X-44", "X-45", *_SEATED_NO_CASH]),
    "bag_opened": (["X-13", "X-14"], ["X-01", "X-02", "X-03", "X-05", "X-34", "X-44", "X-45", *_SEATED_NO_CASH]),
    "machine_count": (["X-01", "X-02", "X-03", "X-04", "X-05", "X-08", "X-10", "X-14", "X-18", "X-19", "X-20",
                       "X-21", "X-34", "X-35", "X-37", "X-38", "X-39", "X-43", "X-44", "X-45", "X-46", "X-47",
                       "X-48"],
                      ["X-13", "X-15", "X-16", "X-17", *_SEATED_NO_CASH]),
    "cash_to_visitor": (["X-01"], ["X-02", "X-03", "X-05", "X-16", "X-17", "X-18", "X-19", "X-34", "X-35", "X-37",
                                   "X-38", "X-39", "X-48", "X-43", "X-44", "X-45"]),
    "detector_check": (["X-35", "X-43", "X-45"], ["X-01", "X-02", "X-03", "X-05", "X-34", "X-37", "X-44", "X-13",
                                                  "X-14", *_SEATED_NO_CASH]),
    "rejects_merged": (["X-44", "X-45"], ["X-02", "X-03", "X-05", "X-34", "X-37", "X-13", "X-14",
                                          *_SEATED_NO_CASH]),
    "writing": (["X-04", "X-05", "X-08", "X-14", "X-19", "X-47", "X-48"],
                ["X-13", "X-34", "X-35", "X-37", "X-38", "X-02", "X-03", "X-17", "X-20", "X-21"]),
    "money_to_safe": (["X-05", "X-08", "X-13", "X-14", "X-19", "X-27", "X-35", "X-39", "X-48"],
                      ["X-01", "X-02", "X-03", "X-04", "X-10", "X-26", "X-34", "X-37", "X-38", "X-43", "X-44",
                       "X-45", "X-46", "X-15", "X-16", "X-17", "X-47", "X-11", "X-12", "X-28", "X-29", "X-30",
                       "X-31", "X-32", "X-33", "X-40"]),
    "hand_count": (["X-43", "X-45"], ["X-44", "X-02", "X-03", "X-34", "X-37", *_SEATED_NO_CASH]),
}


def run(args) -> None:
    from pipeline.video import decode

    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    clips = args.clips.split(",") if args.clips else ALL
    client = VlmClient(concurrency=args.concurrency)
    cfg = {"fps": args.fps, "win_sec": args.win, "stride_sec": args.stride, "n_frames": args.n,
           "scale": args.scale, "crop": args.crop}
    y0, y1 = (float(v) for v in args.crop.split(","))

    def one(clip: str) -> str:
        frames, times, meta = decode(f"{args.videos}/{clip}.mov", args.fps)
        roi = desk_roi(frames[0], y0, y1) if frames else None
        t0 = time.time()
        wf = video_facts(client, frames, times, roi=roi, n_frames=args.n, scale=args.scale, win_sec=args.win,
                         stride_sec=args.stride)
        dt = time.time() - t0
        (out / f"{clip}.json").write_text(json.dumps(
            {"clip": clip, "config": cfg, "duration": meta.duration, "runtime_sec": round(dt, 2),
             "windows": wf}, ensure_ascii=False, indent=1), encoding="utf-8")
        return f"{clip} {meta.duration:.0f}s {len(wf)} windows {dt:.1f}s"

    with ThreadPoolExecutor(args.parallel) as ex:
        for line in ex.map(one, clips):
            print(line, flush=True)


def _first(sf: dict, name: str):
    return sf[name]["t"] if sf[name]["value"] else None


def report(args) -> None:
    out = pathlib.Path(args.out).expanduser()
    data = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("X-*.json"))}
    gt = json.loads(pathlib.Path(args.gt).read_text(encoding="utf-8"))
    clip_facts = {c: session_facts(d["windows"], -1, 1e9, args.votes) for c, d in data.items()}

    print(f"# {out} ({len(data)} clips), votes >= {args.votes}")
    cfgs = {json.dumps(d["config"], sort_keys=True) for d in data.values()}
    print("config:", "; ".join(cfgs))
    rts = sorted((d["runtime_sec"], d["duration"], c) for c, d in data.items())
    print("runtime: max %.1f s (%s, %.0f s clip), mean %.1f s" % (rts[-1][0], rts[-1][2], rts[-1][1],
                                                                 sum(r[0] for r in rts) / len(rts)))
    print("\n## Clip-level facts vs narratives (TP/P = recall on clips with the action, FP/N on clips without)")
    print("| fact | TP/P | FP/N | missed | false alarms |")
    print("|---|---|---|---|---|")
    for name in FACT_NAMES:
        pos, neg = EXPECTED.get(name, ([], []))
        pos = [c for c in pos if c in clip_facts]
        neg = [c for c in neg if c in clip_facts]
        tp = [c for c in pos if clip_facts[c][name]["value"]]
        fp = [c for c in neg if clip_facts[c][name]["value"]]
        print(f"| {name} | {len(tp)}/{len(pos)} | {len(fp)}/{len(neg)} | "
              f"{','.join(c for c in pos if c not in tp)} | {','.join(fp)} |")

    print("\n## Fact x clip (first time s, votes); * = narrative shows it, - = narrative rules it out")
    print("| clip | team | " + " | ".join(FACT_NAMES) + " |")
    print("|---|---|" + "---|" * len(FACT_NAMES))
    for c, sf in clip_facts.items():
        g = gt.get(c, {})
        cells = []
        for name in FACT_NAMES:
            pos, neg = EXPECTED.get(name, ([], []))
            mark = "*" if c in pos else "-" if c in neg else ""
            v = sf[name]
            cells.append(f"{mark}{v['t']:.0f}({v['votes']})" if v["value"] else f"{mark}.")
        print(f"| {c} | {g.get('verdict', '?')[:4]} {','.join(g.get('types', []))} {g.get('t') or ''} | "
              + " | ".join(cells) + " |")

    print("\n## Derived contrasts")
    def row(label, clips, fn):
        print(f"- {label}: " + "; ".join(f"{c}={fn(clip_facts[c])}" for c in clips if c in clip_facts))
    row("T1 handovers within 10 s of the first (1 = money only, >=2 = card + money)  viol X-02,17,19,34,35,37,38 | "
        "clean X-01,03,04,05,16,18,39,47,48",
        ["X-02", "X-17", "X-19", "X-34", "X-35", "X-37", "X-38", "X-01", "X-03", "X-04", "X-05", "X-16", "X-18",
         "X-39", "X-47", "X-48"],
        lambda s: f"{s['handovers_first10']} (hand {','.join(f'{t:g}' for t in s['handovers'][:4])})")
    row("T10 recorded (typing >= 2 obs or writing >= 2 obs within 2.5 s)  viol X-13,34,35,37,38 | "
        "clean X-04,05,08,14,19,47,48",
        ["X-13", "X-34", "X-35", "X-37", "X-38", "X-04", "X-05", "X-08", "X-14", "X-19", "X-47", "X-48"],
        lambda s: s["recorded"])
    row("T13 bag: machine_votes (>= 2 = counted)  viol X-13 | clean X-14", ["X-13", "X-14"],
        lambda s: s["machine_votes"])
    row("T11 cash_to_visitor  viol X-01 | clean X-03", ["X-01", "X-03"], lambda s: _first(s, "cash_to_visitor"))
    row("T13 rejects_merged, detector_check, hand_count  viol X-44 | clean X-45, X-43", ["X-44", "X-45", "X-43"],
        lambda s: f"{_first(s, 'rejects_merged')},{_first(s, 'detector_check')},{_first(s, 'hand_count')}")
    row("T13 bag: bag_opened, machine_count, money_to_safe  viol X-13 | clean X-14", ["X-13", "X-14"],
        lambda s: f"{_first(s, 'bag_opened')},{_first(s, 'machine_count')},{_first(s, 'money_to_safe')}")
    row("T10 writing fact (>= votes windows)  viol X-13,34,35,37,38 | clean X-05,14,19,48",
        ["X-13", "X-34", "X-35", "X-37", "X-38", "X-05", "X-14", "X-19", "X-48"], lambda s: _first(s, "writing"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--out", default="~/forte/cache/facts")
    ap.add_argument("--clips", default="")
    ap.add_argument("--videos", default=str(pathlib.Path("~/forte/videos").expanduser()))
    ap.add_argument("--gt", default=str(pathlib.Path(__file__).resolve().parent.parent / "analysis/gt_team.json"))
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--win", type=float, default=8.0)
    ap.add_argument("--stride", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--crop", default="0.1,0.7", help="y0,y1 fractions of the frame height kept (full width)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--votes", type=int, default=2)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
