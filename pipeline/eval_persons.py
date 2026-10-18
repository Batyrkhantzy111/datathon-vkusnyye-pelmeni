"""Evaluate pipeline.persons on every clip and cache the outputs.

GPU host, from ~/forte:
  ~/det-venv/bin/python -m pipeline.eval_persons [--clips X-11,X-18] [--fps 5] [--out ~/forte/cache/persons]
                                                [--no-vlm] [--from-cache]
Writes OUT/X-NN.json (PersonTracker.run output + timing) and prints:
  1. crowd alarms per clip (detector candidates, VLM-confirmed alarms) - expected only on X-18 (~16 s, label 20 s);
  2. cashier_stand / cashier_leave vs labels on the leave clips;
  3. visitor enter/exit on X-01..X-05;
  4. runtime per clip (decode, detection+tracking, VLM).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline.video import decode  # noqa: E402

VIDEOS = pathlib.Path.home() / "forte/videos"
CROWD_CLIP = "X-18"
LEAVE_LABELS = {"X-11": 18, "X-13": 47, "X-15": 15, "X-22": 14, "X-24": 8, "X-26": 9, "X-28": 7,
                "X-12": None, "X-14": None, "X-23": None, "X-25": None, "X-27": None, "X-29": None}
VISITOR_CLIPS = ["X-01", "X-02", "X-03", "X-04", "X-05"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out", default=str(pathlib.Path.home() / "forte/cache/persons"))
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--from-cache", action="store_true", help="only print the tables from cached outputs")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = args.clips.split(",") if args.clips else sorted(p.stem for p in VIDEOS.glob("X-*.mov"))

    res = {}
    if args.from_cache:
        for c in clips:
            res[c] = json.loads((out / f"{c}.json").read_text())
    else:
        from pipeline.persons import PersonTracker
        t0 = time.time()
        pt = PersonTracker(vlm_verify=not args.no_vlm)
        print(f"model load {time.time() - t0:.1f} s", flush=True)
        for c in clips:
            t0 = time.time()
            frames, times, meta = decode(str(VIDEOS / f"{c}.mov"), fps=args.fps, max_side=960, pix_fmt="bgr24")
            t_dec = time.time() - t0
            r = pt.run(frames, times)
            r["meta"].update({"clip": c, "fps_sampled": args.fps, "video_duration": meta.duration,
                              "decode_sec": round(t_dec, 2), "wall_sec": round(time.time() - t0, 2)})
            (out / f"{c}.json").write_text(json.dumps(r, ensure_ascii=False))
            res[c] = r
            ev = [(e["type"], e["t"]) for e in r["events"]]
            print(f"{c} dur {meta.duration:6.1f}s  wall {r['meta']['wall_sec']:5.2f}s  crowd {r['crowd_intervals']}"
                  f"  events {ev}", flush=True)
    report(res)


def _ev(r, typ):
    return [e["t"] for e in r["events"] if e["type"] == typ]


def report(res: dict):
    print("\n## Crowd (>2 in room)")
    print("| clip | detector candidates [t0-t1 max_n] | VLM counts | alarms | expected |")
    print("|---|---|---|---|---|")
    false_clips = []
    for c, r in res.items():
        cand = r.get("crowd_checks", [])
        exp = "alarm ~16-22 s" if c == CROWD_CLIP else "none"
        if c != CROWD_CLIP and r["crowd_intervals"]:
            false_clips.append(c)
        if cand or c == CROWD_CLIP:
            cs = ", ".join(f"{k['t0']}-{k['t1']} n{k['max_n']}" for k in cand) or "-"
            print(f"| {c} | {cs} | {[k.get('vlm_counts') for k in cand]} | {r['crowd_intervals'] or '-'} | {exp} |")
    print(f"false-alarm clips: {false_clips or 'none'} (of {len(res) - (CROWD_CLIP in res)} clips without a crowd)")

    print("\n## Cashier stand / leave")
    print("| clip | label s | cashier_stand | walk_out | near_exit | cashier_leave | stand-label | leave-label | "
          "returns |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c, lab in LEAVE_LABELS.items():
        if c not in res:
            continue
        r = res[c]
        st, lv, rt = _ev(r, "cashier_stand"), _ev(r, "cashier_leave"), _ev(r, "cashier_return")
        last = [e for e in r["events"] if e["type"] == "cashier_leave"][-1:] or [{}]
        d_st = f"{st[-1] - lab:+.1f}" if st and lab is not None else "-"
        d_lv = f"{lv[-1] - lab:+.1f}" if lv and lab is not None else "-"
        print(f"| {c} | {lab if lab is not None else 'clean'} | {st} | {last[0].get('t_walk_out')} | "
              f"{last[0].get('t_near_exit')} | {lv} | {d_st} | {d_lv} | {rt} |")

    print("\n## Visitors")
    print("| clip | visitor_enter | visitor_exit | max n_room | tracks (id role t_start-t_end) |")
    print("|---|---|---|---|---|")
    for c in VISITOR_CLIPS:
        if c not in res:
            continue
        r = res[c]
        trs = "; ".join(f"{t['id']} {t['role']} {t['t_start']}-{t['t_end']}" for t in r["tracks"]
                        if t["role"] != "unknown")
        print(f"| {c} | {_ev(r, 'visitor_enter')} | {_ev(r, 'visitor_exit')} | "
              f"{max((f['n_room'] for f in r['frames']), default=0)} | {trs} |")

    print("\n## Runtime")
    rows = sorted(res.items(), key=lambda kv: -kv[1]["meta"].get("video_duration", 0))
    tot_v = sum(r["meta"].get("video_duration", 0) for _, r in rows)
    tot_w = sum(r["meta"].get("wall_sec", 0) for _, r in rows)
    for c, r in rows[:5]:
        m = r["meta"]
        print(f"{c}: video {m.get('video_duration', 0):.1f} s, frames {m['n_frames']}, decode {m.get('decode_sec')} s, "
              f"detect {m['detect_sec']} s, run total {m['total_sec']} s, wall {m.get('wall_sec')} s")
    print(f"all {len(rows)} clips: video {tot_v:.0f} s, wall {tot_w:.1f} s")


if __name__ == "__main__":
    main()
