"""Per-frame scene states from Qwen3-VL-8B (local vLLM, structured JSON output).

Two passes per video:
1. layout: the model localizes the desk, safe, laptop, counting machine and door on a few frames; median boxes.
   Nothing is tied to fixed pixel positions, so a new camera angle only moves the boxes;
2. states: per sampled frame, four small requests, each with one image and one focused question:
   - whole frame: people in the room, cashier posture, door;
   - safe crop: safe door open/closed, key in the lock;
   - laptop crop: what the screen shows;
   - desk crop: cash, bag, notebook, ID, clothing, cup/food on the desk.
   Focused crops measured better than one request with all fields (probe on reference frames, pipeline/probe_states.py).
Violation rules never go into the prompts: the model only describes what it sees, the rules engine decides.

CLI (GPU host, from ~/forte):
  ~/det-venv/bin/python -m pipeline.states --clips X-20,X-21|all [--fps 1] [--out ~/forte/cache/states]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
from PIL import Image

from pipeline.vlm import VlmClient, jpeg_b64

LAYOUT_LABELS = ["desk", "safe", "laptop", "counting_machine", "door"]
LAYOUT_PROMPT = (
    "This is a frame from a fixed overhead CCTV camera in a bank cash room. Locate these objects: "
    "desk (the cashier's work desk), safe (the metal safe for cash: a metal cabinet standing on the floor, "
    "usually next to the desk), laptop, counting_machine (the banknote counting machine on the desk), "
    "door (the entrance door of this room). Give one box per object; skip objects that are not visible. "
    "Coordinates are relative to the image on a 0-1000 scale."
)
LAYOUT_SCHEMA = {
    "type": "array",
    "maxItems": 8,
    "items": {
        "type": "object",
        "properties": {
            "label": {"enum": LAYOUT_LABELS},
            "bbox_2d": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 4, "maxItems": 4},
        },
        "required": ["label", "bbox_2d"],
        "additionalProperties": False,
    },
}

SYSTEM = ("You analyse images from a fixed overhead CCTV camera in a bank cash room. Report only what is clearly "
          "visible; when unsure, answer false, none or not_visible.")


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


B = {"type": "boolean"}
QUESTIONS = {
    "room": {
        "crop": None,
        "prompt": (
            "The whole frame of the cash room: a cashier's desk, a metal safe, an entrance door, a glass wall that "
            "can show people outside the room and reflections. Answer as JSON:\n"
            "- people_in_room: number of people physically inside the room (exclude people behind the glass wall "
            "and reflections)\n"
            "- cashier: the employee who works at the desk: seated | standing | absent\n"
            "- door: the entrance door of the room: open (the door leaf is swung open and the doorway is passable) "
            "| closed (the door leaf fills the doorway) | not_visible\n"
            "- id_document: an ID card or passport visible in someone's hand or on the desk\n"
            "- cash_in_hands: banknotes held in someone's hand"),
        "schema": _obj({"people_in_room": {"type": "integer", "minimum": 0, "maximum": 9},
                        "cashier": {"enum": ["seated", "standing", "absent"]},
                        "door": {"enum": ["open", "closed", "not_visible"]},
                        "id_document": B, "cash_in_hands": B}),
    },
    "safe": {
        "crop": (("safe",), 0.25, 0.20, 0.10, 448),
        "prompt": (
            "A zoomed view of the metal safe in a bank cash room. Answer as JSON:\n"
            "- safe: open (the safe door is swung open and the dark inside of the safe is visible) | closed (the "
            "front is a flat closed panel) | not_visible\n"
            "- key_in_safe_lock: a key or key ring is inserted in the lock of the safe door"),
        "schema": _obj({"safe": {"enum": ["open", "closed", "not_visible"]}, "key_in_safe_lock": B}),
    },
    "laptop": {
        "crop": (("laptop",), 0.25, 0.25, 0.15, 384),
        "prompt": (
            "A zoomed view of a laptop on a desk. Answer as JSON:\n"
            "- laptop: white_window (a bright screen showing an application window, a spreadsheet, a document or "
            "a web page) | picture (a screen showing a photo or picture: wallpaper, lock or sign-in screen) | off "
            "(dark screen, lid open) | lid_closed | not_visible"),
        "schema": _obj({"laptop": {"enum": ["white_window", "picture", "off", "lid_closed", "not_visible"]}}),
    },
    "desk": {
        "crop": (("desk", "laptop", "counting_machine"), 0.08, 0.45, 0.05, 672),
        "prompt": (
            "A zoomed view of a cashier's desk in a bank cash room (a banknote counting machine, a laptop, a phone "
            "may stand on it). Answer as JSON with booleans:\n"
            "- cash_on_desk: banknotes or a bundle of banknotes lying on the desk (not in a hand, not inside the "
            "counting machine)\n"
            "- cash_in_hands: banknotes held in someone's hand\n"
            "- cash_in_machine: banknotes inside the counting machine\n"
            "- bag_on_desk: a bag lying on the desk: a cash-collection bag, a fabric or plastic money bag (often "
            "green or grey) or a handbag\n"
            "- notebook_on_desk: a notebook, ledger, folder, binder or papers lying on the desk (a flat rectangular "
            "object of any colour)\n"
            "- id_document: an ID card or passport in a hand or on the desk\n"
            "- clothing_on_desk: a jacket, coat or other clothing lying on the desk\n"
            "- cup_or_food: a cup, mug, can, bottle, food or a jar of sweets on the desk"),
        "schema": _obj({k: B for k in ["cash_on_desk", "cash_in_hands", "cash_in_machine", "bag_on_desk",
                                       "notebook_on_desk", "id_document", "clothing_on_desk", "cup_or_food"]}),
    },
}
# merged per-frame fields (later questions overwrite earlier ones for shared keys, except OR-merged booleans)
OR_FIELDS = {"id_document", "cash_in_hands"}
FIELDS = ["people_in_room", "cashier", "door", "safe", "laptop", "cash_on_desk", "cash_in_hands", "cash_in_machine",
          "bag_on_desk", "notebook_on_desk", "id_document", "clothing_on_desk", "cup_or_food", "key_in_safe_lock"]


def _median_box(boxes: list[list[float]]) -> list[float]:
    return [float(np.median([b[i] for b in boxes])) for i in range(4)]


def layout(client: VlmClient, frames_rgb: list[np.ndarray], n: int = 5) -> dict:
    """{label: [x1, y1, x2, y2] in pixels of the given frames}, median over up to n evenly spaced frames."""
    if not frames_rgb:
        return {}
    idx = sorted({int(round(i)) for i in np.linspace(0, len(frames_rgb) - 1, min(n, len(frames_rgb)))})
    h, w = frames_rgb[0].shape[:2]
    reqs = [{"prompt": LAYOUT_PROMPT, "images": [jpeg_b64(frames_rgb[i], scale=1.5)], "schema": LAYOUT_SCHEMA,
             "max_tokens": 300} for i in idx]
    found: dict[str, list[list[float]]] = {}
    for ans in client.ask_many(reqs):
        try:
            items = json.loads(ans) if ans else []
        except json.JSONDecodeError:
            continue
        seen = set()
        for it in items:
            lab, bb = it.get("label"), it.get("bbox_2d")
            if lab not in LAYOUT_LABELS or lab in seen or not bb or len(bb) != 4:
                continue
            x1, x2 = sorted((bb[0], bb[2]))
            y1, y2 = sorted((bb[1], bb[3]))
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            seen.add(lab)
            found.setdefault(lab, []).append([x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h])
    need = max(1, len(idx) // 2)
    return {lab: _median_box(bs) for lab, bs in found.items() if len(bs) >= need}


def crop_box(lay: dict, spec: tuple, w: int, h: int) -> tuple | None:
    """Pixel crop box (x1, y1, x2, y2) around the union of the layout boxes named in spec, or None."""
    labels, ex, et, eb, _side = spec
    bs = [lay[lab] for lab in labels if lab in lay]
    if not bs:
        return None
    x1, y1 = min(b[0] for b in bs), min(b[1] for b in bs)
    x2, y2 = max(b[2] for b in bs), max(b[3] for b in bs)
    bw, bh = x2 - x1, y2 - y1
    box = (int(max(0, x1 - ex * bw)), int(max(0, y1 - et * bh)), int(min(w, x2 + ex * bw)), int(min(h, y2 + eb * bh)))
    return box if box[2] - box[0] >= 16 and box[3] - box[1] >= 16 else None


def crop_b64(frame: np.ndarray, box: tuple, side: int) -> str:
    x1, y1, x2, y2 = box
    im = Image.fromarray(np.ascontiguousarray(frame[y1:y2, x1:x2]))
    s = side / max(im.width, im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    return jpeg_b64(np.asarray(im))


def frame_states(client: VlmClient, frames_rgb: list[np.ndarray], times: list[float], lay: dict) -> list[dict]:
    """[{t, ok, **FIELDS}] in frame order. Fields of a failed or skipped question are absent; ok=False if the
    whole-frame question failed."""
    if not frames_rgb:
        return []
    h, w = frames_rgb[0].shape[:2]
    reqs, keys = [], []
    for i, f in enumerate(frames_rgb):
        for name, q in QUESTIONS.items():
            if q["crop"] is None:
                img = jpeg_b64(f, scale=1.25)
            else:
                box = crop_box(lay, q["crop"], w, h)
                if box is None:
                    continue
                img = crop_b64(f, box, q["crop"][4])
            reqs.append({"system": SYSTEM, "prompt": q["prompt"], "images": [img], "schema": q["schema"],
                         "max_tokens": 120})
            keys.append((i, name))
    out = [{"t": t, "ok": False} for t in times]
    for (i, name), ans in zip(keys, client.ask_many(reqs)):
        if not ans:
            continue
        try:
            d = json.loads(ans)
        except json.JSONDecodeError:
            continue
        rec = out[i]
        for k, v in d.items():
            rec[k] = (rec.get(k, False) or v) if k in OR_FIELDS else v
        if name == "room":
            rec["ok"] = True
    return out


ABBR = {"cashier": {"seated": "S", "standing": "U", "absent": "-"},
        "door": {"open": "O", "closed": ".", "not_visible": "?"},
        "safe": {"open": "O", "closed": ".", "not_visible": "?"},
        "laptop": {"white_window": "W", "picture": "P", "off": "o", "lid_closed": "c", "not_visible": "?"}}


def timeline(states: list[dict]) -> str:
    """Compact one-row-per-field view for eyeballing a clip."""
    rows = []
    for f in FIELDS:
        cells = []
        for s in states:
            v = s.get(f)
            if v is None:
                cells.append("x")
            elif f == "people_in_room":
                cells.append(str(v))
            elif f in ABBR:
                cells.append(ABBR[f].get(v, "?"))
            else:
                cells.append("#" if v else ".")
        rows.append(f"{f[:16]:>16} {''.join(cells)}")
    ticks = "".join(str(int(round(s["t"])) // 10 % 10) if abs(s["t"] - round(s["t"] / 10) * 10) < 1e-6 else " "
                    for s in states)
    return "\n".join([f"{'t (x10 s)':>16} {ticks}"] + rows)


def to_grid(states: list[dict], grid: list[float]) -> list[dict]:
    """States resampled onto a denser time grid by nearest sample (the rules count frames at 1 fps; long videos
    are sampled sparser to save VLM time)."""
    if not states:
        return []
    ts = np.array([s["t"] for s in states])
    return [dict(states[int(np.argmin(np.abs(ts - t)))], t=float(t)) for t in grid]


def main():
    from pipeline.video import decode

    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--videos", default=str(pathlib.Path.home() / "forte/videos"))
    ap.add_argument("--out", default=str(pathlib.Path.home() / "forte/cache/states"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = (sorted(p.stem for p in pathlib.Path(a.videos).glob("X-*.mov")) if a.clips == "all"
             else a.clips.split(","))
    client = VlmClient()
    for clip in clips:
        t0 = time.time()
        frames, times, meta = decode(f"{a.videos}/{clip}.mov", a.fps)
        t1 = time.time()
        lay = layout(client, frames)
        t2 = time.time()
        states = frame_states(client, frames, times, lay)
        t3 = time.time()
        (out / f"{clip}.json").write_text(json.dumps(
            {"clip": clip, "fps": a.fps, "duration": meta.duration, "size": [frames[0].shape[1], frames[0].shape[0]],
             "layout": lay, "states": states}, ensure_ascii=False), encoding="utf-8")
        bad = sum(not s["ok"] for s in states)
        lay_s = " ".join(f"{k}=[{','.join(str(int(v)) for v in b)}]" for k, b in lay.items())
        print(f"== {clip} {meta.duration:.1f}s frames={len(frames)} decode={t1 - t0:.1f}s layout={t2 - t1:.1f}s "
              f"states={t3 - t2:.1f}s failed={bad} | {lay_s}", flush=True)
        print(timeline(states), flush=True)


if __name__ == "__main__":
    main()
