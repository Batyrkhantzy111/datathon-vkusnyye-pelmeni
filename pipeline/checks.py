"""Checks around two events: the cashier leaving the room and someone entering through the door.

Single-frame scene questions miss small or ambiguous things (a notebook or a cash bag on the desk, the key being
taken from the safe, a shut vs open glass door). Here every question is about one event window:
- leave_check: the empty room a few seconds after the cashier walked out is the evidence of what stayed behind
  (one narrow question per object on a desk zoom); the laptop screen is compared before/after; taking the safe key
  is asked on a numbered sequence of safe zooms (two overlapping windows before t_leave, either one counts).
- entry_check: the door state is NOT asked from the VLM: on this camera Qwen3-VL-8B answers "open" for a shut glass
  door (single crops, before/after pairs and full frames were all tried, see eval_checks). Instead the door crop is
  compared pixel-wise with a reference frame taken before the entry, when the door was shut: the door counts as
  closed again once the difference falls back to the reference level and stays there. Whether another person came
  in or watched through the doorway is asked from the VLM on a numbered sequence of the upper part of the frame.
Prompts ask only what is visible; the rules engine decides what is a violation.

API (times in seconds; frames_rgb/times from any decode of the clip; measured at 1 fps and 2 fps, same results
except took_key, e.g. pipeline.video.decode):
    leave_check(client, frames_rgb, times, t_stand, t_leave, layout=None) -> dict
    entry_check(client, frames_rgb, times, t_enter, layout=None) -> dict
layout: {desk, safe, laptop, counting_machine, door: [x1, y1, x2, y2]} in pixels of frames_rgb
(pipeline.states.layout); without it the checks fall back to fixed fractions of the frame.
"""
from __future__ import annotations

import json
import time

import numpy as np
from PIL import Image, ImageDraw

from pipeline.vlm import VlmClient, jpeg_b64
from pipeline.vlm_facts import _font

SYSTEM = (
    "You look at frames from a fixed CCTV camera in a bank cash room. The room has a cashier's desk with a "
    "banknote counting machine and a laptop, a light-grey metal safe standing next to the desk, and a glass "
    "entrance door. A glass wall can show people outside the room and reflections. Report only what is clearly "
    "visible."
)

# fallback boxes as fractions of the frame (the portrait 464x848 camera positions seen so far)
_DEFAULT = {"desk": (0.05, 0.42, 0.75, 0.67), "safe": (0.42, 0.30, 0.62, 0.44), "laptop": (0.48, 0.37, 0.68, 0.47),
            "door": (0.0, 0.02, 0.28, 0.53)}


def _box(layout: dict | None, name: str, w: int, h: int) -> list[float]:
    if layout and name in layout:
        return list(layout[name])
    f = _DEFAULT[name]
    return [f[0] * w, f[1] * h, f[2] * w, f[3] * h]


def _expand(b: list[float], w: int, h: int, lr: float, top: float, bottom: float) -> tuple[int, int, int, int]:
    """Box grown by fractions of its size (negative = shrunk), clipped to the frame."""
    bw, bh = b[2] - b[0], b[3] - b[1]
    return (int(max(0, b[0] - lr * bw)), int(max(0, b[1] - top * bh)), int(min(w, b[2] + lr * bw)),
            int(min(h, b[3] + bottom * bh)))


def _union(bs: list[list[float]]) -> list[float]:
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def _img(frame: np.ndarray, box: tuple | None, side: int, label: str | None = None) -> str:
    """Crop (or the whole frame), resize to `side` px on the long side, draw `label` in the top-left corner."""
    im = Image.fromarray(np.ascontiguousarray(frame if box is None else frame[box[1]:box[3], box[0]:box[2]]))
    s = side / max(im.width, im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    if label:
        d = ImageDraw.Draw(im)
        size = max(16, im.height // 20)
        f = _font(size)
        x1, y1, x2, y2 = d.textbbox((0, 0), label, font=f)
        d.rectangle((0, 0, x2 - x1 + size // 2, y2 - y1 + size // 2), fill=(0, 0, 0))
        d.text((size // 4 - x1, size // 4 - y1), label, fill=(255, 255, 0), font=f)
    return jpeg_b64(np.asarray(im), quality=90)


def _at(times: list[float], t: float) -> int:
    """Index of the frame nearest to t."""
    return int(np.argmin(np.abs(np.asarray(times) - t)))


def _span(times: list[float], t0: float, t1: float, n: int) -> list[int]:
    """Up to n distinct frame indices evenly spread over [t0, t1]."""
    idx = [i for i, t in enumerate(times) if t0 - 1e-6 <= t <= t1 + 1e-6] or [_at(times, (t0 + t1) / 2)]
    if len(idx) > n:
        idx = [idx[int(round(j))] for j in np.linspace(0, len(idx) - 1, n)]
    return idx


def _step(times: list[float], idx: list[int]) -> float:
    return round((times[idx[-1]] - times[idx[0]]) / max(1, len(idx) - 1), 1)


def _json(ans: str | None) -> dict:
    if not ans:
        return {}
    try:
        obj = json.loads(ans)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


YES_NO = {"enum": ["yes", "no"]}

# ---------------------------------------------------------------- leave

DESK_ITEMS = {
    "cash": "banknotes: a bundle, a stack or single notes of paper money (small light rectangles of paper)",
    "bag": "a cash-collection bag or money sack (fabric or plastic, often green or grey)",
    "notebook_or_documents": "a notebook, ledger, folder, tablet-like dark book, or sheets of paper",
    "cup_or_food": "a cup, mug, can, bottle, jar or food",
    "clothing": "a jacket, coat, scarf or handbag",
}
_DESK_SCHEMA = _obj({"where": {"type": "string", "maxLength": 120}, "present": {"type": "boolean"}})


def _desk_prompt(item: str) -> str:
    return (f"This is a zoom on the cashier's desk top. Is there {item} lying on the desk top? "
            "Write `where` (where on the desk it lies, or 'none') then `present`.")


LAPTOP_SCREEN = ["white_window", "picture", "off", "lid_closed", "not_visible"]
_LAPTOP_PROMPT = (
    "The images are zooms on the laptop on the desk: image 1 while the cashier still works, image 2 after she "
    "left the room.\nFirst write `screen_now`: what the laptop screen shows in image 2 (colours, content).\n"
    "Then `screen`: what image 2 shows: white_window = a bright white or light application window "
    "(spreadsheet, document, browser page); picture = a dark-blue or colourful full-screen picture or wallpaper, "
    "with or without a clock (lock or sign-in screen); off = the lid is open but the screen is black or dark; "
    "lid_closed = the laptop lid is shut, no screen visible; not_visible = the laptop cannot be seen."
)
_LAPTOP_SCHEMA = _obj({"screen_now": {"type": "string", "maxLength": 200}, "screen": {"enum": LAPTOP_SCREEN}})


def _key_prompt(n: int, step: float) -> str:
    return (f"The {n} images are zooms on the metal safe, {step:g} s apart, numbered 1-{n}. The cashier is at the "
            "safe door, then walks away. A small key sits in the keyhole near the left edge of the safe door. Write "
            "`hands`: what her hand does at the keyhole in these frames. Then `key_in_hand`: she pulls the key out "
            "of the keyhole and walks away holding it. `safe_door_last`: the safe door in the last image.")


_KEY_SCHEMA = _obj({"hands": {"type": "string", "maxLength": 200}, "key_in_hand": YES_NO,
                    "safe_door_last": {"enum": ["shut", "open"]}})


def _joint_prompt(n: int, step: float) -> str:
    items = "; ".join(f"`{k}` = {v}" for k, v in DESK_ITEMS.items())
    return (
        f"Images 1-{n} are zooms on the metal safe, in time order, {step:g} s apart (numbered in the corner): the "
        "cashier is at the safe door, then walks out of the room. A small key sits in the keyhole near the left "
        f"edge of the safe door. Image {n + 1} (D) is a zoom on her desk top after she left. Image {n + 2} (L1) is "
        f"the laptop while she still worked, image {n + 3} (L2) the laptop after she left. Image {n + 4} (K) is a "
        "strongly magnified crop of the safe keyhole after she left.\n"
        "Answer as JSON:\n"
        "- `hands`: what her hand does at the keyhole in images 1-{n}; `key_in_hand`: yes if she pulls the key out "
        "of the keyhole and walks away holding it; `safe_door_last`: the safe door in the last safe image.\n"
        f"- `desk`: objects lying on the desk top in image D; then for image D, true only if it clearly lies on the "
        f"desk top: {items}.\n"
        "- `screen_now`: what the laptop screen shows in L2; `screen`: white_window = a bright white or light "
        "application window; picture = a dark-blue or colourful full-screen picture or wallpaper (lock or sign-in "
        "screen); off = lid open, screen black; lid_closed = lid shut; not_visible.\n"
        "- `keyhole`: in image K: only_dark_dot = just a small dark dot or slot; key_or_object_attached = a key, "
        "a key head, a ring or another object sticks out of or hangs from the keyhole."
    ).replace("{n}", str(n))


_JOINT_SCHEMA = _obj({
    "hands": {"type": "string", "maxLength": 200}, "key_in_hand": YES_NO,
    "safe_door_last": {"enum": ["shut", "open"]},
    "desk": {"type": "string", "maxLength": 200}, **{k: {"type": "boolean"} for k in DESK_ITEMS},
    "screen_now": {"type": "string", "maxLength": 160}, "screen": {"enum": LAPTOP_SCREEN},
    "keyhole": {"enum": ["only_dark_dot", "key_or_object_attached"]},
})

# ---- key left in the safe lock (T5): pixel cue, the VLM does not see it (analysis/key_probe/)
# A key left in the lock hangs on a cord straight down from the keyhole: a thin darker streak on the light-grey door
# just below the dark keyhole dot. Offsets are for 848-px-high frames (the open clips); other sizes are rescaled.
KEY_CORD_MAX = -4.5  # % darkness of the streak vs the door; median <= this before AND after -> key stays in the lock
KEY_DOOR_LIT = 140.0  # grey level of the door next to the keyhole; darker = light off / occluded -> no reading
_KEY_REF_H = 848.0
_RING = ((-3, 0), (3, 0), (0, -3), (0, 3), (-3, -3), (3, 3), (-3, 3), (3, -3))


def _key_cue(frame: np.ndarray, safe: list[float]) -> tuple[float, float, float] | None:
    """(cue %, keyhole x, y in frame pixels) or None when the keyhole is not readable (hidden, dark room).
    Keyhole = the darkest small dot (vs its 8 neighbours 3 px away, all on lit metal) in the left part of the safe
    door; cue = darkest 1-px column of a 5x10 px strip just below it, relative to the door, in %."""
    H, W = frame.shape[:2]
    s = _KEY_REF_H / H
    x1, y1, x2, y2 = (v * s for v in safe)
    w, h = x2 - x1, y2 - y1
    cx0, cy0 = int(max(0, x1 - 20)), int(max(0, y1))
    cx1, cy1 = int(min(W * s, x2)), int(min(H * s, y2 + 20))
    if cx1 - cx0 < 20 or cy1 - cy0 < 20:
        return None
    im = Image.fromarray(np.ascontiguousarray(frame[int(cy0 / s):int(np.ceil(cy1 / s)),
                                                    int(cx0 / s):int(np.ceil(cx1 / s))])).convert("L")
    if abs(s - 1.0) > 0.02:
        im = im.resize((cx1 - cx0, cy1 - cy0), Image.BILINEAR)
    L = np.asarray(im, dtype=np.float32)
    p = np.pad(L, 1, mode="edge")
    g = sum(p[dy:dy + L.shape[0], dx:dx + L.shape[1]] for dy in range(3) for dx in range(3)) / 9.0
    X0, X1 = max(9, int(x1 - cx0 + 0.03 * w)), min(L.shape[1] - 9, int(x1 - cx0 + 0.22 * w))
    Y0, Y1 = max(9, int(0.25 * h)), min(L.shape[0] - 14, int(0.65 * h))
    if X1 <= X0 or Y1 <= Y0:
        return None
    ring = np.stack([g[Y0 + dy:Y1 + dy, X0 + dx:X1 + dx] for dy, dx in _RING])
    score = np.median(ring, 0) - g[Y0:Y1, X0:X1]
    score[ring.min(0) < 0.85 * KEY_DOOR_LIT] = -1e9
    yy, xx = np.unravel_index(int(np.argmax(score)), score.shape)
    if score[yy, xx] < 20:
        return None
    x, y = X0 + int(xx), Y0 + int(yy)
    door = float(np.median(L[y - 8:y + 12, x + 3:x + 9]))
    if door < KEY_DOOR_LIT:
        return None
    col = L[y + 3:y + 13, x - 2:x + 3].mean(0)
    return round(float((col.min() - door) / door * 100), 1), (cx0 + x) / s, (cy0 + y) / s


def _key_reads(frames_rgb: list[np.ndarray], times: list[float], safe: list[float], t0: float, t1: float) -> list:
    return [(float(t), _key_cue(f, safe)) for f, t in zip(frames_rgb, times) if t0 - 1e-6 <= t <= t1 + 1e-6]


def key_left_check(frames_rgb: list[np.ndarray], times: list[float], t_stand: float, t_leave: float,
                   safe: list[float]) -> dict:
    """Key left in the safe lock: the key cord under the keyhole is seen while she works (median over
    [clip start, t_stand - 0.5]) AND after she walked out (median over [t_leave + 0.5, t_leave + 3], else the last 2
    frames after t_leave; >= 2 readable frames). Taken key = seen before, gone after. t_lock = end of the last stretch
    in [t_stand, t_leave] where the keyhole is hidden (her hand at the lock): when she is done with the lock."""
    before = [r[0] for _, r in _key_reads(frames_rgb, times, safe, times[0], t_stand - 0.5) if r]
    after = _key_reads(frames_rgb, times, safe, t_leave + 0.5, t_leave + 3.0)
    if len(after) < 2:
        after = [(float(t), _key_cue(f, safe)) for f, t in list(zip(frames_rgb, times))[-2:] if t > t_leave]
    seen = [(t, r) for t, r in after if r]
    cue_b = float(np.median(before)) if before else None
    cue_a = float(np.median([r[0] for _, r in seen])) if len(seen) >= 2 else None
    mid = _key_reads(frames_rgb, times, safe, t_stand, t_leave)
    t_lock = next((mid[i][0] for i in range(len(mid) - 2, -1, -1) if mid[i][1] is None and mid[i + 1][1]), None)
    hole = seen[-1][1][1:] if seen else None
    return {"key_in_lock": cue_b is not None and cue_a is not None and max(cue_b, cue_a) <= KEY_CORD_MAX,
            "cue_before": cue_b, "cue_after": cue_a, "t_lock": t_lock, "frames_after": [t for t, _ in seen],
            "lock_box": [round(hole[0] - 40, 1), round(hole[1] - 40, 1), round(hole[0] + 40, 1),
                         round(hole[1] + 60, 1)] if hole else None}


def leave_check(client: VlmClient, frames_rgb: list[np.ndarray], times: list[float], t_stand: float,
                t_leave: float, layout: dict | None = None, after_sec: float = 3.0, mode: str = "split") -> dict:
    """What the cashier left behind when she stood up at t_stand and walked out at t_leave.

    mode="split" (default, the measured-best variant): concurrent requests, one per desk object on the desk zoom
    after_sec after t_leave; laptop screen (zoom before t_stand vs after leaving); taking the safe key (safe zooms
    over [t_leave - 7, t_leave - 2.5] and [t_leave - 4.5, t_leave], <= 10 frames each, yes in either window); key
    still in the lock: pixel cue key_left_check (no VLM request; Qwen3-VL-8B never sees the key).
    mode="joint": ONE request with the safe sequence [t_leave - 7, t_leave] (<= 10 frames) + desk zoom after +
    laptop zooms before/after + keyhole crop and one combined schema.
    evidence_t is the 'after' frame (empty room). raw["failed"] lists the parts whose request or JSON failed;
    their fields then hold the clean default (false / not_visible / closed) and must not be read as observations.
    """
    h, w = frames_rgb[0].shape[:2]
    t0_clip, t_end = times[0], times[-1]
    i_after = _at(times, min(t_end, t_leave + after_sec))
    i_before = _at(times, max(t0_clip, t_stand - 1.0))
    desk = _expand(_union([_box(layout, "desk", w, h), _box(layout, "laptop", w, h)]), w, h, 0.05, 0.30, -0.25)
    lap = _expand(_box(layout, "laptop", w, h), w, h, 0.35, 0.35, 0.15)
    sb = _box(layout, "safe", w, h)
    safe = _expand(sb, w, h, 0.6, 0.4, 0.2)
    bw, bh = sb[2] - sb[0], sb[3] - sb[1]
    lock = (int(max(0, sb[0] - 0.2 * bw)), int(sb[1] + 0.345 * bh), int(sb[0] + 0.25 * bw), int(sb[1] + 0.695 * bh))
    after_desk = _img(frames_rgb[i_after], desk, 896)
    lap_imgs = [_img(frames_rgb[i_before], lap, 384, "L1"), _img(frames_rgb[i_after], lap, 384, "L2")]

    if mode == "joint":
        idx = _span(times, max(t0_clip, t_leave - 7.0), t_leave, 10)
        key_windows = [idx]
        req = {"system": SYSTEM, "prompt": _joint_prompt(len(idx), _step(times, idx)), "schema": _JOINT_SCHEMA,
               "max_tokens": 450,
               "images": [_img(frames_rgb[i], safe, 384, str(k + 1)) for k, i in enumerate(idx)]
               + [_img(frames_rgb[i_after], desk, 896, "D"), *lap_imgs, _img(frames_rgb[i_after], lock, 448, "K")]}
        t0 = time.time()
        a = _json(client.ask_many([req])[0])
        sec = round(time.time() - t0, 2)
        desk_ans = {k: {"present": a[k]} for k in DESK_ITEMS if k in a}
        a_lap = {"screen": a["screen"], "screen_now": a.get("screen_now")} if "screen" in a else {}
        a_keys = [{k: a[k] for k in ("hands", "key_in_hand", "safe_door_last") if k in a}] if a else [{}]
        a_lock = {"answer": a["keyhole"]} if "keyhole" in a else {}
        ok = {"desk": bool(desk_ans), "laptop": bool(a_lap), "key": bool(a_keys[0]), "lock": bool(a_lock)}
        raw_extra = {"joint": a}
    else:
        key_windows = [_span(times, max(t0_clip, a), max(t0_clip, b), 10)
                       for a, b in ((t_leave - 7.0, t_leave - 2.5), (t_leave - 4.5, t_leave))]
        reqs = [{"system": SYSTEM, "prompt": _desk_prompt(v), "schema": _DESK_SCHEMA, "max_tokens": 80,
                 "images": [after_desk]} for v in DESK_ITEMS.values()]
        reqs.append({"system": SYSTEM, "prompt": _LAPTOP_PROMPT, "schema": _LAPTOP_SCHEMA, "max_tokens": 120,
                     "images": lap_imgs})
        for idx in key_windows:
            reqs.append({"system": SYSTEM, "prompt": _key_prompt(len(idx), _step(times, idx)),
                         "schema": _KEY_SCHEMA, "max_tokens": 170,
                         "images": [_img(frames_rgb[i], safe, 384, str(k + 1)) for k, i in enumerate(idx)]})
        t0 = time.time()
        answers = [_json(x) for x in client.ask_many(reqs)]
        sec = round(time.time() - t0, 2)
        n_items = len(DESK_ITEMS)
        desk_ans = {k: x for k, x in zip(DESK_ITEMS, answers[:n_items]) if x}
        a_lap = answers[n_items]
        a_keys = answers[n_items + 1:n_items + 3]
        a_lock = {}
        ok = {"desk": len(desk_ans) == n_items, "laptop": bool(a_lap), "key": all(a_keys)}
        raw_extra = {}
    key_left = key_left_check(frames_rgb, times, t_stand, t_leave, sb)
    return {
        "left_on_desk": {k: bool(desk_ans.get(k, {}).get("present", False)) for k in DESK_ITEMS},
        "laptop_screen_after": a_lap.get("screen", "not_visible"),
        "took_key_from_safe": any(x.get("key_in_hand") == "yes" for x in a_keys),
        "key_left_in_safe": key_left["key_in_lock"],
        "t_lock": key_left["t_lock"],
        "lock_box": key_left["lock_box"],
        "safe_closed": a_keys[-1].get("safe_door_last", "shut") == "shut",
        "evidence_t": float(times[i_after]),
        "raw": {"mode": mode, "desk": desk_ans, "laptop": a_lap, "key_windows": a_keys, "lock": a_lock,
                "key_left": key_left, **raw_extra,
                "frames": {"before": float(times[i_before]), "after": float(times[i_after]),
                           "key_windows": [[float(times[i]) for i in idx] for idx in key_windows]},
                "failed": [k for k, v in ok.items() if not v], "vlm_sec": sec},
    }


# ---------------------------------------------------------------- entry

DOOR_DIFF_CLOSED = 15.0  # mean abs grey difference vs the reference (0-255) below which the door counts as shut


def _door_vec(frame: np.ndarray, box: tuple) -> np.ndarray:
    """Upper 45% of the door crop (the door leaf above the desk and people), grey 48x96, mean removed."""
    x1, y1, x2, y2 = box
    crop = frame[y1:y1 + max(2, int(0.45 * (y2 - y1))), x1:x2]
    a = np.asarray(Image.fromarray(np.ascontiguousarray(crop)).convert("L").resize((48, 96), Image.BOX), np.float32)
    return a - a.mean()


def _people_prompt(n: int, step: float) -> str:
    return (f"The {n} images are frames of one video in time order, about {step:g} s apart, numbered 1-{n}. In "
            "image 1 a person (the newcomer) comes in through the entrance door on the left.\nWrite `summary`: one "
            "sentence on every person who comes through the doorway or stands in it over these frames. Then "
            "`other_person`: yes only if, after the newcomer, a SECOND, different person comes in through the "
            "door, or stands in the doorway or right outside the door looking in. The newcomer, a person who was "
            "already in the room in image 1 (for example the employee seated at the desk) and people passing far "
            "behind the glass wall do not count. `frame`: first image with that second person at the door, 0 if "
            "none.")


def _people_schema(n: int) -> dict:
    return _obj({"summary": {"type": "string", "maxLength": 200}, "other_person": YES_NO,
                 "frame": {"type": "integer", "minimum": 0, "maximum": n}})


def entry_check(client: VlmClient, frames_rgb: list[np.ndarray], times: list[float], t_enter: float,
                layout: dict | None = None, after_sec: float = 10.0, other_sec: float = 15.0,
                pre_sec: float = 3.0, stable_sec: float = 1.5, t_ref: float | None = None) -> dict:
    """Whether the door is shut again after someone came in at t_enter, and whether anyone else followed.

    Door: reference = the frame pre_sec before t_enter (the door is assumed shut then), or the frame at t_ref when
    given (the caller passes a moment with a visitor seated inside when this entry follows another visitor's exit:
    then the door is in use right before t_enter). Per frame in [ref, t_enter + after_sec] the door-crop
    difference to the reference; the door opened at the first frame above DOOR_DIFF_CLOSED; it counts as closed
    again at the start of the first run of frames below the threshold, at or after t_enter, lasting >= stable_sec.
    No frame above the threshold = the door never moved although someone came through: it already stood open, so
    not closed. Limit: if the door stood open at the reference and the newcomer shuts it, the crop differs from the
    (open) reference to the end and the result is a false 'not closed'.
    Other person: one VLM request, 8 frames over [t_enter, t_enter + other_sec] of the upper 55% of the frame.
    """
    h, w = frames_rgb[0].shape[:2]
    t_end = times[-1]
    door = _expand(_box(layout, "door", w, h), w, h, 0.0, 0.0, 0.0)
    i_ref = _at(times, max(times[0], t_enter - pre_sec) if t_ref is None else t_ref)
    i_last = _at(times, min(t_end, t_enter + after_sec))
    ref = _door_vec(frames_rgb[i_ref], door)
    diffs = [(float(times[i]), float(np.abs(_door_vec(frames_rgb[i], door) - ref).mean()))
             for i in range(i_ref, i_last + 1)]
    opened = next((k for k, (_, d) in enumerate(diffs) if d >= DOOR_DIFF_CLOSED), None)
    # nobody passes a shut door without moving it: no change vs the reference means it already stood open
    t_opened, t_closed, closed = None, None, False
    if opened is not None:
        t_opened = diffs[opened][0]
        # the door must be shut again after the newcomer is in (a reference taken before a previous person's exit
        # can show the door shut for a moment between that exit and this entry)
        start = opened
        while start < len(diffs) and diffs[start][0] < t_enter:
            start += 1
        run_start = None
        for t, d in diffs[start:]:
            if d < DOOR_DIFF_CLOSED:
                run_start = t if run_start is None else run_start
                if t - run_start >= stable_sec - 1e-6:
                    t_closed, closed = run_start, True
                    break
            else:
                run_start = None

    top = (0, 0, w, int(h * 0.55))
    ppl = _span(times, t_enter, min(t_end, t_enter + other_sec), 8)
    req = {"system": SYSTEM, "prompt": _people_prompt(len(ppl), _step(times, ppl)), "schema": _people_schema(len(ppl)),
           "max_tokens": 200, "images": [_img(frames_rgb[i], top, 448, str(k + 1)) for k, i in enumerate(ppl)]}
    t0 = time.time()
    a_ppl = _json(client.ask_many([req])[0])
    sec = round(time.time() - t0, 2)
    other = a_ppl.get("other_person") == "yes"
    k = a_ppl.get("frame")
    t_other = float(times[ppl[k - 1]]) if other and isinstance(k, int) and 1 <= k <= len(ppl) else None
    return {
        "door_closed_after_entry": closed,
        "t_closed": t_closed,
        "someone_else_entered_or_watched": other,
        "t_other": t_other,
        "evidence_t": float(t_closed if t_closed is not None else times[i_last]),
        "raw": {"ref_t": float(times[i_ref]), "t_opened": t_opened, "door_box": list(door),
                "door_diff": [[round(t, 2), round(d, 1)] for t, d in diffs], "people": a_ppl,
                "people_frames": [float(times[i]) for i in ppl], "ok": bool(a_ppl), "vlm_sec": sec},
    }
