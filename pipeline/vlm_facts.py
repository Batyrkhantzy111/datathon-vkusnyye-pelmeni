"""Transactional facts over short time windows from Qwen3-VL-8B (local vLLM, structured JSON output).

Per-frame states cannot tell who handed what to whom or in which order. Here the model sees a window of 8
consecutive frames (2 fps, 4 s step, each frame numbered in its corner) and, for each event in a small group,
gives the number of the first frame where the event is visible or 0. Asking for a frame number is the least
yes-biased form we found: yes/no questions that name an action (also all facts in one request, or an event log
over a closed vocabulary) pulled the model towards "yes" on clean clips. Prompts describe only what to look
for, never violation rules (the model parrots rules instead of looking, MODEL_SHORTLIST.md §4).

Measured on the 42 open clips (pipeline/eval_facts.py, cache ~/forte/cache/facts): see the module report.
Reliable: visitor->cashier handovers (structure of the visit), machine counting, money into the safe,
typing/writing, bag opening. Not reliable: telling an ID card from a banknote stack in a single frame,
the authenticity detector, merging rejected notes (X-44 vs X-45).

API:
    video_facts(client, frames_rgb, times, windows=None, roi=None) -> per window
        {t_start, t_end, ok, summary, timeline: {event: t | None}, facts: {fact: {"value": bool, "t": float | None}}}
    session_facts(window_facts, t_enter, t_exit, min_votes=2) -> {fact: {value, t, votes}, handovers, ...}
    desk_roi(frame) -> default crop (full width, 10-70 % of the height: desk, people, safe, door; not the floor)
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.vlm import VlmClient, jpeg_b64

SYSTEM = (
    "You watch short sequences of frames from a fixed overhead CCTV camera in a bank cash room. A cashier sits "
    "at a desk with a banknote counting machine, a small authenticity detector and a laptop; a metal safe "
    "stands next to the desk; visitors sit on the other side of the desk. A glass wall can show people outside "
    "the room and reflections. Report only actions you can actually see in the frames. When unsure, answer 0."
)

# One request per group and window; the window images are a shared prefix that vLLM prefix caching reuses.
EVENT_GROUPS: list[dict[str, str]] = [
    {"visitor_gives": "the visitor (the person on the other side of the desk) passes any object into the "
                      "cashier's hand",
     "card_examined": "the cashier holds a small card (identity card) up in front of her and looks at it for a "
                      "moment",
     "money_back_to_visitor": "the cashier passes banknotes into the visitor's hand"},
    {"band_removed": "the cashier removes a rubber band or a paper band from a stack of banknotes",
     "into_machine": "the cashier puts banknotes into the banknote counting machine",
     "detector_check": "the cashier holds single banknotes against the small authenticity detector (a small "
                       "device with a lamp, separate from the counting machine)",
     "rejects_merged": "the cashier puts a few separate banknotes (set aside earlier, or taken from the "
                       "machine's reject pocket) onto the main stack of banknotes",
     "hand_count": "the cashier counts banknotes by hand, moving the notes one by one between her fingers"},
    {"bag_opened": "the cashier cuts or unzips a cash bag and takes money out of it",
     "into_safe": "the cashier puts money into the safe",
     "writing": "the cashier writes in a notebook with a pen",
     "typing": "the cashier types on the laptop keyboard"},
]
EVENTS = [k for g in EVENT_GROUPS for k in g]

# event -> fact (a fact is true in a window when any of its events is)
EVENT_FACT = {"visitor_gives": "cash_received", "card_examined": "id_document",
              "money_back_to_visitor": "cash_to_visitor", "band_removed": "machine_count",
              "into_machine": "machine_count", "detector_check": "detector_check",
              "rejects_merged": "rejects_merged", "hand_count": "hand_count", "bag_opened": "bag_opened",
              "into_safe": "money_to_safe", "writing": "writing", "typing": "writing"}
FACT_NAMES = list(dict.fromkeys(EVENT_FACT.values()))


def _head(n: int, step: float) -> str:
    return (f"The {n} images are consecutive frames of one video, {step:g} s apart, numbered 1-{n} in the "
            "top-left corner.")


def _schema(group: dict, n: int) -> dict:
    return {"type": "object",
            "properties": {"what_happens": {"type": "string", "maxLength": 240},
                           **{f"frame_{k}": {"type": "integer", "minimum": 0, "maximum": n} for k in group}},
            "required": ["what_happens", *(f"frame_{k}" for k in group)], "additionalProperties": False}


def _prompt(n: int, step: float, group: dict) -> str:
    lines = [_head(n, step),
             "Write `what_happens`: one sentence on what the cashier's and the visitor's hands do in these frames.",
             "Then give the number of the first frame where each event is visible, or 0 if it does not happen "
             "in these frames:"]
    lines += [f"- frame_{k}: {v}" for k, v in group.items()]
    return "\n".join(lines)


@lru_cache(maxsize=4)
def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _tile(frame: np.ndarray, num: int, roi: tuple | None, scale: float) -> str:
    """Crop, scale, draw the frame number in the top-left corner, encode as JPEG base64."""
    im = Image.fromarray(np.ascontiguousarray(frame if roi is None else frame[roi[1]:roi[3], roi[0]:roi[2]]))
    if scale != 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BICUBIC)
    d = ImageDraw.Draw(im)
    size = max(18, im.height // 18)
    f = _font(size)
    txt = str(num)
    x1, y1, x2, y2 = d.textbbox((0, 0), txt, font=f)
    d.rectangle((0, 0, x2 - x1 + size // 2, y2 - y1 + size // 2), fill=(0, 0, 0))
    d.text((size // 4 - x1, size // 4 - y1), txt, fill=(255, 255, 0), font=f)
    return jpeg_b64(np.asarray(im), quality=88)


def desk_roi(frame: np.ndarray, y0: float = 0.1, y1: float = 0.7) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    return 0, int(h * y0), w, int(h * y1)


def make_windows(times: list[float], win_sec: float = 8.0, stride_sec: float = 4.0) -> list[tuple[float, float]]:
    """Half-open [t_start, t_end) windows covering the video; the last one is aligned to the end."""
    if not times:
        return []
    t0, t1 = times[0], times[-1] + 1e-6
    if t1 - t0 <= win_sec:
        return [(t0, t1)]
    out, s = [], t0
    while s + win_sec < t1:
        out.append((s, s + win_sec))
        s += stride_sec
    out.append((max(t0, t1 - win_sec), t1))
    return out


def _pick(times: list[float], t_start: float, t_end: float, n: int) -> list[int]:
    idx = [i for i, t in enumerate(times) if t_start <= t < t_end]
    if len(idx) <= n:
        return idx
    return [idx[int(round(j))] for j in np.linspace(0, len(idx) - 1, n)]


def video_facts(client: VlmClient, frames_rgb: list[np.ndarray], times: list[float],
                windows: list[tuple[float, float]] | None = None, roi: tuple | None = None,
                n_frames: int = 8, scale: float = 1.0, win_sec: float = 8.0, stride_sec: float = 4.0
                ) -> list[dict]:
    """Facts per time window, in window order.

    frames_rgb/times: a 2 fps decode (pipeline.video.decode(path, 2.0)); each window takes up to `n_frames`
    evenly spaced frames from it.
    windows: [(t_start, t_end)] s; default sliding `win_sec` windows with a `stride_sec` step. The stride matters:
    with 6 s the handover structure (session_facts) became unstable on 3 of 16 visit clips.
    roi: (x1, y1, x2, y2) crop applied to every frame; default desk_roi(frame).
    Returns [{t_start, t_end, ok, summary, timeline: {event: t | None}, facts: {fact: {value, t}}}];
    ok=False where a request or its JSON failed (its events are None).
    """
    if windows is None:
        windows = make_windows(times, win_sec, stride_sec)
    if roi is None and frames_rgb:
        roi = desk_roi(frames_rgb[0])
    reqs, picks = [], []
    for ts, te in windows:
        idx = _pick(times, ts, te, n_frames)
        picks.append(idx)
        if not idx:
            continue
        step = round((times[idx[-1]] - times[idx[0]]) / max(1, len(idx) - 1), 2)
        imgs = [_tile(frames_rgb[i], k + 1, roi, scale) for k, i in enumerate(idx)]
        reqs += [{"system": SYSTEM, "prompt": _prompt(len(idx), step, g), "images": imgs,
                  "schema": _schema(g, len(idx)), "max_tokens": 160} for g in EVENT_GROUPS]
    answers = iter(client.ask_many(reqs) if reqs else [])
    out = []
    for (ts, te), idx in zip(windows, picks):
        tl: dict[str, float | None] = {k: None for k in EVENTS}
        ok, notes = bool(idx), []
        for g in EVENT_GROUPS if idx else []:
            ans = next(answers)
            try:
                obj = json.loads(ans) if ans else None
            except json.JSONDecodeError:
                obj = None
            if not isinstance(obj, dict):
                ok = False
                continue
            notes.append(str(obj.get("what_happens", "")))
            for k in g:
                f = obj.get(f"frame_{k}")
                if isinstance(f, int) and 1 <= f <= len(idx):
                    tl[k] = float(times[idx[f - 1]])
        facts = {name: {"value": False, "t": None} for name in FACT_NAMES}
        for k, t in tl.items():
            f = facts[EVENT_FACT[k]]
            if t is not None and (f["t"] is None or t < f["t"]):
                facts[EVENT_FACT[k]] = {"value": True, "t": t}
        out.append({"t_start": float(ts), "t_end": float(te), "ok": ok, "summary": " | ".join(notes),
                    "timeline": tl, "facts": facts})
    return out


def _times(window_facts: list[dict], event: str, t_enter: float, t_exit: float) -> list[float]:
    return sorted(t for w in window_facts for t in [(w.get("timeline") or {}).get(event)]
                  if t is not None and t_enter <= t <= t_exit)


def clusters(ts: list[float], gap: float = 2.5) -> list[list[float]]:
    """Group sorted times into runs whose neighbours are at most `gap` s apart."""
    out: list[list[float]] = []
    for t in sorted(ts):
        if out and t - out[-1][-1] <= gap:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def session_facts(window_facts: list[dict], t_enter: float, t_exit: float, min_votes: int = 2) -> dict:
    """Aggregate window facts over a visitor session [t_enter, t_exit] (use -inf, inf for the whole video).

    Per fact: value = seen in at least `min_votes` windows with the fact time inside the session (overlapping
    windows see the same event twice, so 2 filters one-off hallucinations); t = earliest such time; votes.

    Visit structure:
    - "handovers": start times of separate visitor->cashier handovers (observations clustered, 2.5 s gap);
    - "first_handover": time of the first handover (the cashier first takes something), None if none;
    - "handovers_first10": handovers starting within 10 s of the first one. The model cannot tell an ID card
      from a banknote stack at this resolution, but the visit structure shows it: with an ID check the visitor
      hands over twice (card, then money) in quick succession; a single handover means money was taken with no
      card first. Open clips: 1 in all 7 T1 clips; >= 2 in 7 of 9 clips with an ID check (X-18, X-39 give 1).
    - "recorded": typing seen >= 2 times, or writing seen >= 2 times within 2.5 s (single hits are noise);
    - "machine_votes": observations of notes going into the counting machine (X-13 uncounted bag: 1; X-14: 17);
    - "timeline": {event: sorted observation times inside the session}, for evidence frames and custom rules.
    """
    out: dict = {}
    for name in FACT_NAMES:
        ts = []
        for w in window_facts:
            f = w["facts"].get(name) or {}
            if f.get("value") and f.get("t") is not None and t_enter <= f["t"] <= t_exit:
                ts.append(f["t"])
        hit = len(ts) >= min_votes
        out[name] = {"value": hit, "t": min(ts) if hit else None, "votes": len(ts)}
    tl = {k: _times(window_facts, k, t_enter, t_exit) for k in EVENTS}
    hand = [c[0] for c in clusters(tl["visitor_gives"])]
    out["handovers"] = hand
    out["first_handover"] = hand[0] if hand else None
    out["handovers_first10"] = len([t for t in hand if t <= hand[0] + 10.0]) if hand else 0
    out["recorded"] = len(tl["typing"]) >= 2 or any(len(c) >= 2 for c in clusters(tl["writing"], 2.5))
    out["machine_votes"] = len(tl["into_machine"])
    out["timeline"] = tl
    return out
