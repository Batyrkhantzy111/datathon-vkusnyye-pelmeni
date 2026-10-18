"""Video analysis entry point used by the service: Analyzer.analyze(video) -> result dict (spec §10 + extras).

Stages:
1. decode: 2 fps RGB frames (transaction facts use all of them, scene states every other one = 1 fps; sparser for
   very long videos) and 5 fps BGR frames for the person tracker;
2. in parallel: person tracking (YOLO26x + tracker, GPU), VLM layout + per-frame scene states, VLM transaction
   facts over sliding windows (Qwen3-VL-8B on the local vLLM server);
3. rules (pipeline/rules.py) -> candidates; norms registry (pipeline/norms.json) -> type name, description, rule;
4. evidence frames at the exact timestamps, with the relevant region outlined.
Everything runs on this host; nothing is sent to external services.
"""
from __future__ import annotations

import base64
import importlib
import io
import json
import logging
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from pipeline import rules, states
from pipeline.video import decode, frames_at, probe
from pipeline.vlm import VlmClient

log = logging.getLogger("pipeline.analyze")
NORMS = json.loads((pathlib.Path(__file__).with_name("norms.json")).read_text(encoding="utf-8"))
VLM_NAME = "Qwen3-VL-8B-Instruct (BF16, локальный сервер vLLM)"
DET_NAME = "YOLO26x (детекция людей) + трекинг"
MAX_STATE_FRAMES = 200  # longer videos get sparser scene-state sampling


def _optional(module: str):
    try:
        return importlib.import_module(module)
    except Exception as e:  # a missing or broken optional stage must not take the service down
        log.warning("optional module %s unavailable: %s", module, e)
        return None


def _jpeg_b64(img: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Analyzer:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.vlm = VlmClient()
        if not self._wait_vlm(300):
            raise RuntimeError("локальный сервер VLM (vLLM, 127.0.0.1:8001) недоступен")
        self.persons_mod = _optional("pipeline.persons")
        self.tracker = None
        if self.persons_mod is not None:
            try:
                self.tracker = self.persons_mod.PersonTracker(device=device)
            except Exception as e:
                log.warning("PersonTracker init failed: %s", e)
        self.facts_mod = _optional("pipeline.vlm_facts")
        self.checks_mod = _optional("pipeline.checks")

    def _wait_vlm(self, timeout: float) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.vlm.ready():
                return True
            time.sleep(3)
        return False

    # ---------------------------------------------------------------- stages
    def _persons(self, path: str, lay_future) -> dict | None:
        if self.tracker is None:
            return None
        frames, times, _ = decode(path, 5.0, max_side=960, pix_fmt="bgr24")
        # foot line for "in the room" vs "behind the glass / reflection": the open-footage prior, moved up to just
        # above the desk when the VLM desk box says the desk is higher in the frame (tilted / closer camera;
        # robustness test analysis/robust/report.md, fix 1: 0 changes on the 42 open clips)
        gf = None
        try:
            lay = lay_future.result()
            if "desk" in lay and lay.get("_h"):
                gf = min(self.persons_mod.GLASS_FRAC, lay["desk"][1] / lay["_h"] - 0.02)
        except Exception as e:
            log.warning("layout for the glass line unavailable: %s", e)
        return self.tracker.run(frames, times, glass_frac=gf)

    def _layout(self, frames: list[np.ndarray]) -> dict:
        lay = states.layout(self.vlm, frames)
        lay["_h"] = frames[0].shape[0]
        return lay

    def _states(self, frames: list[np.ndarray], times: list[float], lay_future) -> list[dict]:
        lay = {k: v for k, v in lay_future.result().items() if not k.startswith("_")}
        return states.frame_states(self.vlm, frames, times, lay)

    def _facts(self, frames: list[np.ndarray], times: list[float]) -> list[dict] | None:
        if self.facts_mod is None:
            return None
        return self.facts_mod.video_facts(self.vlm, frames, times)

    def _checks(self, frames: list[np.ndarray], times: list[float], lay: dict, sig: rules.Signals) -> dict:
        """Multi-frame checks around each cashier leave and each entry through the door (run concurrently)."""
        if self.checks_mod is None:
            return {}
        lay = {k: v for k, v in lay.items() if not k.startswith("_")}
        leaves = rules.find_leaves(sig, rules.PARAMS)
        ents = [t for t, _ in rules.entries(sig) if sig.duration - t >= rules.PARAMS["door_min_after"]][:4]
        out: dict = {"leave": [], "entry": []}

        def leave(lv):
            r = self.checks_mod.leave_check(self.vlm, frames, times, lv.t_stand, lv.t_leave, layout=lay)
            return dict(r, t_leave=lv.t_leave, t_stand=lv.t_stand)

        def entry(t):
            r = self.checks_mod.entry_check(self.vlm, frames, times, t, layout=lay, t_ref=rules.door_ref_time(sig, t))
            return dict(r, t_enter=t)

        with ThreadPoolExecutor(max_workers=4) as ex:
            fl = [ex.submit(leave, lv) for lv in leaves]
            fe = [ex.submit(entry, t) for t in ents]
            for kind, futs in (("leave", fl), ("entry", fe)):
                for f in futs:
                    try:
                        out[kind].append(f.result())
                    except Exception as e:
                        log.warning("%s check failed: %s", kind, e)
        return out

    # ---------------------------------------------------------------- output
    def _evidence(self, path: str, cand: dict, lay: dict, persons: dict | None) -> list[dict]:
        ts = [round(float(cand["t"]), 2)]
        for t in sorted(cand.get("evidence_t") or []):
            if t is not None and all(abs(t - u) >= 1.0 for u in ts) and len(ts) < 3:
                ts.append(round(float(t), 2))
        ts.sort()
        out = []
        for t, fr in zip(ts, frames_at(path, ts)):
            img = Image.fromarray(np.ascontiguousarray(fr))
            d = ImageDraw.Draw(img)
            reg = cand.get("region")
            if reg in ("desk", "safe", "laptop", "door") and reg in lay:
                d.rectangle(lay[reg], outline=(255, 40, 40), width=3)
            elif reg == "persons" and persons:
                fp = min(persons.get("frames") or [], key=lambda f: abs(f["t"] - t), default=None)
                for bx in (fp or {}).get("boxes") or []:
                    if len(bx) < 7 or bx[6] == "room":
                        d.rectangle(bx[:4], outline=(255, 40, 40), width=3)
            zb = cand.get("zoom")
            if zb:  # magnified inset of a small detail (T5: the safe lock), pasted into the top-right corner
                x1, y1, x2, y2 = (int(round(v)) for v in zb)
                d.rectangle([x1, y1, x2, y2], outline=(255, 200, 0), width=2)
                z = img.crop((x1, y1, x2, y2))
                k = max(1.0, min(4.0, 0.5 * img.width / max(1, z.width)))
                z = z.resize((round(z.width * k), round(z.height * k)), Image.LANCZOS)
                img.paste(z, (img.width - z.width, 0))
                d.rectangle([img.width - z.width, 0, img.width - 1, z.height - 1], outline=(255, 200, 0), width=3)
            label = f" {rules.mmss(t)} "
            d.rectangle([0, 0, 7 * len(label) + 8, 20], fill=(0, 0, 0))
            d.text((4, 4), label, fill=(255, 255, 0))
            out.append({"timestamp_sec": t, "image_base64": _jpeg_b64(img)})
        return out

    def _violation(self, path: str, cand: dict, lay: dict, persons: dict | None) -> dict:
        ty = NORMS["types"].get(cand["type_id"]) or NORMS["types"]["GENERIC"]
        r = ty["rule"]
        return {
            "type": ty["name"],
            "type_id": cand["type_id"],
            "timecode_sec": round(float(cand["t"]), 1),
            "description": f"{cand['observation']} {ty['why']}",
            "rule": {"document": r["document"], "short": r.get("short"), "clause": r["clause"], "url": r["url"],
                     "quote": r.get("quote"), "strength": ty.get("strength")},
            "confidence": round(float(cand.get("conf", 0.5)), 2),
            "evidence_frames": self._evidence(path, cand, lay, persons),
        }

    # ---------------------------------------------------------------- main
    def analyze(self, video_path: str, workdir: str,
                progress: Callable[[float, str], None] | None = None) -> dict:
        t0 = time.time()
        prog = progress or (lambda p, s: None)
        prog(0.02, "Декодирование видео")
        meta = probe(video_path)
        frames2, times2, _ = decode(video_path, 2.0, max_side=960, pix_fmt="rgb24")
        if not frames2:
            raise ValueError("в видео нет кадров")
        # scene states at 1 fps (0.5 fps measured worse on the open clips: T2 lost on X-04/X-47, T12 false alarms on
        # X-05/X-48); sparser only beyond MAX_STATE_FRAMES frames
        step = max(2, int(np.ceil(len(frames2) / MAX_STATE_FRAMES)))
        frames1, times1 = frames2[::step], times2[::step]
        grid = times2[::2]
        prog(0.08, "Анализ людей, сцены и действий")
        timings = {}

        def timed(name, fn, *a):
            s = time.time()
            try:
                return fn(*a)
            finally:
                timings[name] = round(time.time() - s, 1)

        with ThreadPoolExecutor(max_workers=4) as ex:
            f_layout = ex.submit(timed, "layout", self._layout, frames1)
            f_persons = ex.submit(timed, "persons", self._persons, video_path, f_layout)
            f_facts = ex.submit(timed, "facts", self._facts, frames2, times2)
            f_states = ex.submit(timed, "states", self._states, frames1, times1, f_layout)
            lay = f_layout.result()
            st = f_states.result()
            if step > 2:
                st = states.to_grid(st, grid)
            prog(0.5, "Действия с деньгами и документами")
            try:
                facts = f_facts.result()
            except Exception as e:
                log.warning("facts failed: %s", e)
                facts = None
            prog(0.75, "Люди в кабинете")
            try:
                persons = f_persons.result()
            except Exception as e:
                log.warning("person tracking failed: %s", e)
                persons = None
        sig = rules.Signals(duration=meta.duration, states=st, persons=persons, facts=facts)
        prog(0.8, "Проверка событий: уход кассира, вход в кабинет")
        try:
            sig.checks = timed("checks", self._checks, frames2, times2, lay, sig)
        except Exception as e:
            log.warning("checks failed: %s", e)
        prog(0.88, "Проверка правил")
        cands = rules.evaluate(sig)
        prog(0.9, "Кадры-доказательства")
        violations = [self._violation(video_path, c, lay, persons) for c in cands]
        models = [VLM_NAME] + ([DET_NAME] if persons is not None else [])
        result = {
            "verdict": "violation" if violations else "no_violation",
            "violations": violations,
            "meta": {"duration_sec": round(meta.duration, 2), "fps": round(meta.fps, 2),
                     "frames_analyzed": len(frames2), "processing_sec": round(time.time() - t0, 1),
                     "stage_sec": timings, "models": models},
        }
        try:
            signals = {"duration": meta.duration, "layout": lay, "states": st, "persons": persons, "facts": facts,
                       "checks": sig.checks, "candidates": cands}
            out = pathlib.Path(workdir).parent / "signals.json"
            out.write_text(json.dumps(signals, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception as e:
            log.warning("could not save signals: %s", e)
        prog(0.99, "Готово")
        return result
