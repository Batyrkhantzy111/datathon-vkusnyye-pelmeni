"""Probe dedicated crop questions on reference frames with known states (development tool).

For each (clip, t) it decodes the frame, takes the layout from ~/forte/cache/states/<clip>.json (or computes
it), cuts the safe / door / laptop / desk crops, asks one question per crop, prints the answers and saves the
crops to ~/forte/cache/probe/ for eyeballing.

Usage: ~/det-venv/bin/python -m pipeline.probe_states
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np
from PIL import Image

from pipeline import states as S
from pipeline.vlm import VlmClient, jpeg_b64

HOME = pathlib.Path.home()
CASES = [  # clip, t, expectation
    ("X-20", 8, "safe open"), ("X-21", 12, "safe closed"), ("X-20", 3, "safe open"),
    ("X-22", 20, "key in lock, laptop ?"), ("X-23", 17, "no key, laptop ?"),
    ("X-30", 9, "door open"), ("X-31", 15, "door closed"), ("X-32", 12, "door open"), ("X-33", 12, "door open"),
    ("X-28", 12, "laptop working"), ("X-29", 14, "laptop lock"), ("X-11", 17, "laptop on"),
    ("X-12", 30, "laptop off/closed"),
    ("X-15", 18, "bag on desk"), ("X-26", 14, "cash on desk"), ("X-24", 13, "notebook on desk"),
    ("X-25", 25, "desk clean?"), ("X-40", 18, "jacket on desk"), ("X-46", 12, "cup/food"),
]

Q = {
    "safe": ("This is a zoomed view of a metal safe in a bank cash room. Answer as JSON: door: is the safe door "
             "open (swung open, the dark inside of the safe is visible) or closed (the front is a flat closed "
             "panel); key_in_lock: is a key inserted in the lock of the safe (a small key or key ring sticking "
             "out of the lock)?",
             {"type": "object", "properties": {"door": {"enum": ["open", "closed", "not_visible"]},
                                               "key_in_lock": {"type": "boolean"}},
              "required": ["door", "key_in_lock"], "additionalProperties": False}),
    "door": ("This is a zoomed view of the entrance door of a bank cash room (a glass or solid door in a wall). "
             "Answer as JSON: door: open (the door leaf is swung open and the doorway is passable) or closed "
             "(the door leaf fills the doorway).",
             {"type": "object", "properties": {"door": {"enum": ["open", "closed", "not_visible"]}},
              "required": ["door"], "additionalProperties": False}),
    "laptop": ("This is a zoomed view of a laptop on a desk. Answer as JSON: screen: working (lit screen showing "
               "an application, a document, a table or a desktop with windows) | lock (lit screen showing a lock "
               "or sign-in screen, a full-screen photo or a big clock) | off (dark screen, lid open) | "
               "lid_closed | not_visible.",
               {"type": "object", "properties": {"screen": {"enum": ["working", "lock", "off", "lid_closed",
                                                                     "not_visible"]}},
                "required": ["screen"], "additionalProperties": False}),
    "desk": ("This is a zoomed view of a cashier's desk in a bank cash room. Answer as JSON with booleans: "
             "cash_on_desk (banknotes or a bundle of banknotes lying on the desk, not in a hand and not inside the "
             "counting machine), cash_in_hands (banknotes held in a hand), cash_in_machine (banknotes inside the "
             "banknote counting machine), cash_bag_on_desk (a cash-collection bag or money bag lying on the desk), "
             "notebook_on_desk (a notebook, ledger, folder or papers on the desk), id_document (an ID card or "
             "passport in a hand or on the desk), clothing_on_desk (a jacket, coat, handbag or other clothing on "
             "the desk), cup_or_food (a cup, mug, bottle, food or a jar of sweets on the desk).",
             {"type": "object", "properties": {k: {"type": "boolean"} for k in [
                 "cash_on_desk", "cash_in_hands", "cash_in_machine", "cash_bag_on_desk", "notebook_on_desk",
                 "id_document", "clothing_on_desk", "cup_or_food"]},
              "required": ["cash_on_desk", "cash_in_hands", "cash_in_machine", "cash_bag_on_desk",
                           "notebook_on_desk", "id_document", "clothing_on_desk", "cup_or_food"],
              "additionalProperties": False}),
}
CROP = {  # labels, expand left/right, top, bottom, long side
    "safe": (("safe",), 0.25, 0.20, 0.10, 448),
    "door": (("door",), 0.08, 0.04, 0.04, 448),
    "laptop": (("laptop",), 0.25, 0.25, 0.15, 384),
    "desk": (("desk", "laptop", "counting_machine"), 0.08, 0.45, 0.05, 672),
}


def frame_at(clip: str, t: float) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(HOME / f"forte/videos/{clip}.mov"),
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         check=True, capture_output=True).stdout
    meta = S_probe(clip)
    return np.frombuffer(raw, np.uint8).reshape(meta[1], meta[0], 3)


def S_probe(clip):
    from pipeline.video import probe
    m = probe(str(HOME / f"forte/videos/{clip}.mov"))
    return m.width, m.height


def crop(frame, lay, name):
    labels, ex, et, eb, side = CROP[name]
    bs = [lay[lab] for lab in labels if lab in lay]
    if not bs:
        return None
    h, w = frame.shape[:2]
    x1, y1 = min(b[0] for b in bs), min(b[1] for b in bs)
    x2, y2 = max(b[2] for b in bs), max(b[3] for b in bs)
    bw, bh = x2 - x1, y2 - y1
    box = (int(max(0, x1 - ex * bw)), int(max(0, y1 - et * bh)), int(min(w, x2 + ex * bw)), int(min(h, y2 + eb * bh)))
    im = Image.fromarray(np.ascontiguousarray(frame[box[1]:box[3], box[0]:box[2]]))
    s = side / max(im.width, im.height)
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)


def main():
    client = VlmClient()
    out = HOME / "forte/cache/probe"
    out.mkdir(parents=True, exist_ok=True)
    reqs, keys = [], []
    for clip, t, exp in CASES:
        f = frame_at(clip, t)
        cache = HOME / f"forte/cache/states/{clip}.json"
        lay = (json.loads(cache.read_text()).get("layout") if cache.exists() else None) or S.layout(client, [f])
        for name in Q:
            im = crop(f, lay, name)
            if im is None:
                continue
            im.save(out / f"{clip}_{t}_{name}.jpg", quality=90)
            reqs.append({"prompt": Q[name][0], "schema": Q[name][1], "images": [jpeg_b64(np.asarray(im))],
                         "max_tokens": 120})
            keys.append((clip, t, exp, name))
    answers = client.ask_many(reqs)
    cur = None
    for (clip, t, exp, name), ans in zip(keys, answers):
        if (clip, t) != cur:
            cur = (clip, t)
            print(f"\n== {clip} t={t}  expect: {exp}")
        try:
            ans = json.dumps(json.loads(ans), separators=(",", ":"))
        except (TypeError, json.JSONDecodeError):
            pass
        print(f"   {name:6} {ans}")


if __name__ == "__main__":
    main()
