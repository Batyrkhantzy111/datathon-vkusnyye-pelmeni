"""Evaluate pipeline.checks on the ground-truth clips: leave checks (X-11..X-29) and entry checks (X-30..X-38).

Event times come from ~/forte/cache/persons/X-NN.json events when present, else from the per-frame states
(~/forte/cache/states/X-NN.json). Prints expected vs got per clip and writes ~/forte/cache/checks/X-NN.json.

CLI (GPU host, from ~/forte):
  ~/det-venv/bin/python -m pipeline.eval_checks [--clips X-11,X-30] [--fps 2]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from pipeline.checks import entry_check, leave_check
from pipeline.video import decode
from pipeline.vlm import VlmClient

HOME = pathlib.Path.home() / "forte"

# leave clips: expected violation flags (laptop left working, cash / bag / notebook left on the desk, key left in
# the safe lock) and whether she takes the safe key with her (None = not stated in the labels)
LEAVE = {
    "X-11": ({"laptop_on"}, None), "X-28": ({"laptop_on"}, True), "X-15": ({"bag"}, None),
    "X-26": ({"cash"}, True), "X-24": ({"notebook"}, True), "X-22": ({"key_left"}, False),
    "X-12": (set(), None), "X-23": (set(), True), "X-25": (set(), True), "X-27": (set(), True),
    "X-29": (set(), True), "X-13": (None, None), "X-14": (None, None),
}
# entry clips: (door closed after entry, someone else entered or watched through the doorway)
ENTRY = {
    "X-30": (False, True), "X-31": (True, False), "X-32": (False, False), "X-33": (False, True),
    "X-01": (True, False), "X-02": (True, False), "X-03": (True, False), "X-05": (True, False),
    "X-34": (True, False), "X-35": (True, False), "X-37": (True, False), "X-38": (True, False),
}
FLAGS = ["laptop_on", "cash", "bag", "notebook", "key_left"]
ENTER_EVENTS = ("visitor_enter", "cashier_enter", "cashier_return")


def _load(kind: str, clip: str) -> dict | None:
    p = HOME / "cache" / kind / f"{clip}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _person_event(clip: str, names: tuple[str, ...]) -> float | None:
    """Time of the earliest person-tracker event of one of the given types."""
    d = _load("persons", clip)
    ts = [float(e["t"]) for e in (d or {}).get("events", []) if e.get("type") in names]
    return min(ts) if ts else None


def leave_times(states: list[dict]) -> tuple[float, float] | None:
    """(t_stand, t_leave) when the clip ends with the cashier absent for >= 2 frames, else None (no leave).
    t_leave = start of that final absent run; t_stand = first frame after the last seated frame before it
    (t_leave - 6 s if she never sits)."""
    cs = [s.get("cashier") for s in states]
    ts = [float(s["t"]) for s in states]
    if len(cs) < 2 or cs[-1] != "absent" or cs[-2] != "absent":
        return None
    k_leave = len(cs) - 1
    while k_leave > 0 and cs[k_leave - 1] == "absent":
        k_leave -= 1
    seated = [k for k in range(k_leave) if cs[k] == "seated"]
    t_leave = ts[k_leave]
    t_stand = ts[seated[-1] + 1] if seated else max(ts[0], t_leave - 6.0)
    return t_stand, t_leave


def entry_time(states: list[dict]) -> float:
    """First frame where the cashier turns from absent to present, else where the people count goes up."""
    cs = [s.get("cashier") for s in states]
    if cs and cs[0] == "absent":
        for s, c in zip(states, cs):
            if c != "absent":
                return float(s["t"])
    n0 = states[0].get("people_in_room", 0) if states else 0
    for s in states:
        if (s.get("people_in_room") or 0) > n0:
            return float(s["t"])
    return float(states[0]["t"]) if states else 0.0


def _flags(r: dict) -> set[str]:
    d = r["left_on_desk"]
    out = {k for k, f in (("cash", "cash"), ("bag", "bag"), ("notebook", "notebook_or_documents")) if d[f]}
    if r["laptop_screen_after"] == "white_window":
        out.add("laptop_on")
    if r["key_left_in_safe"]:
        out.add("key_left")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=",".join([*LEAVE, *ENTRY]))
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--times", choices=["persons", "states"], default="persons",
                    help="event times from the person tracker (fallback: states) or from the per-frame states")
    ap.add_argument("--no-layout", action="store_true", help="call the checks without layout boxes (fallback ROIs)")
    ap.add_argument("--mode", choices=["split", "joint"], default="split", help="leave_check request mode")
    ap.add_argument("--out", default=str(HOME / "cache" / "checks"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    client = VlmClient()
    leave_rows, entry_rows = [], []
    for clip in a.clips.split(","):
        st = _load("states", clip)
        lay = None if a.no_layout else (st or {}).get("layout")
        frames, times, _ = decode(str(HOME / "videos" / f"{clip}.mov"), a.fps, max_side=848)
        rec = {"clip": clip, "fps": a.fps, "times_source": a.times, "layout": lay is not None, "mode": a.mode}
        if clip in LEAVE:
            ts = tl = None
            if a.times == "persons":
                ts = _person_event(clip, ("cashier_stand",))
                tl = _person_event(clip, ("cashier_leave",))
            if ts is None or tl is None:
                ts, tl = leave_times(st["states"]) or (None, None)
            if tl is None:
                rec.update(kind="leave", no_leave=True)
                print(f"{clip}: no leave event (tracker and states), leave_check not run", flush=True)
            else:
                t0 = time.time()
                r = leave_check(client, frames, times, ts, tl, layout=lay, mode=a.mode)
                r["wall_sec"] = round(time.time() - t0, 2)
                rec.update(kind="leave", t_stand=ts, t_leave=tl, result=r)
                exp, key = LEAVE[clip]
                got = _flags(r)
                leave_rows.append((clip, ts, tl, exp, got, key, r))
        if clip in ENTRY:
            te = _person_event(clip, ENTER_EVENTS) if a.times == "persons" else None
            if te is None:
                te = entry_time(st["states"])
            t0 = time.time()
            r = entry_check(client, frames, times, te, layout=lay)
            r["wall_sec"] = round(time.time() - t0, 2)
            rec.update(kind="entry", t_enter=te, result=r)
            entry_rows.append((clip, te, ENTRY[clip], r))
        (out / f"{clip}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{clip} done", flush=True)

    if leave_rows:
        print("\n== leave_check  (flags: laptop_on=white_window, cash/bag/notebook left on desk, key_left in lock)")
        print(f"{'clip':5} {'stand':>5} {'leave':>5}  {'expected':22} {'got':22} {'ok':3} {'screen':12} "
              f"{'key exp/got':11} {'cup':3} {'clo':3} {'sec':>5}")
        n_ok = n = 0
        per = {f: [0, 0, 0, 0] for f in FLAGS}  # tp fp fn tn
        key_ok = key_n = 0
        for clip, ts, tl, exp, got, key, r in leave_rows:
            ok = "" if exp is None else ("Y" if exp == got else "N")
            if exp is not None:
                n += 1
                n_ok += exp == got
                for f in FLAGS:
                    per[f][0 if f in exp and f in got else 1 if f in got else 2 if f in exp else 3] += 1
            if key is not None:
                key_n += 1
                key_ok += key == r["took_key_from_safe"]
            e = "-" if exp is None else ",".join(sorted(exp)) or "none"
            print(f"{clip:5} {ts:5.1f} {tl:5.1f}  {e:22} {','.join(sorted(got)) or 'none':22} {ok:3} "
                  f"{r['laptop_screen_after']:12} {str(key)[0]}/{str(r['took_key_from_safe'])[0]:9} "
                  f"{str(r['left_on_desk']['cup_or_food'])[0]:3} {str(r['left_on_desk']['clothing'])[0]:3} "
                  f"{r['raw']['vlm_sec']:5.1f} {','.join(r['raw']['failed'])}")
        print(f"clip-exact {n_ok}/{n} (laptop scored as white_window vs anything else; cup/clothing unscored, no "
              f"labels); took_key {key_ok}/{key_n}")
        for f, (tp, fp, fn, tn) in per.items():
            print(f"  {f:10} tp={tp} fp={fp} fn={fn} tn={tn}")
    if entry_rows:
        print("\n== entry_check")
        print(f"{'clip':5} {'enter':>5}  {'closed exp/got':14} {'t_closed':>8}  {'other exp/got':13} {'sec':>5}  "
              "door diff vs reference (from ref_t, per frame)")
        cd = co = 0
        for clip, te, (ec, eo), r in entry_rows:
            gc, go = r["door_closed_after_entry"], r["someone_else_entered_or_watched"]
            cd += ec == gc
            co += eo == go
            seq = " ".join(f"{d:.0f}" for _, d in r["raw"]["door_diff"])
            tc = "-" if r["t_closed"] is None else f"{r['t_closed']:.1f}"
            print(f"{clip:5} {te:5.1f}  {str(ec)[0]}/{str(gc)[0]} {'Y' if ec == gc else 'N':10} {tc:>8}  "
                  f"{str(eo)[0]}/{str(go)[0]} {'Y' if eo == go else 'N':9} {r['raw']['vlm_sec']:5.1f}  {seq}")
        print(f"door {cd}/{len(entry_rows)}; other person {co}/{len(entry_rows)}")


if __name__ == "__main__":
    main()
