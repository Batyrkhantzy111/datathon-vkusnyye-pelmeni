"""Stand-in analyzer for UI/API testing (FORTE_FAKE=1); the real path is pipeline.analyze.Analyzer.

Sleeps a few seconds, reporting progress, then returns a schema-valid result whose evidence frames are real
JPEG frames cut from the uploaded video.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

from pipeline.video import probe

NORMS = json.loads((Path(__file__).resolve().parent.parent / "pipeline" / "norms.json").read_text(encoding="utf-8"))


def fake_violation(type_id: str, t: float, frames: list[dict], conf: float) -> dict:
    ty = NORMS["types"][type_id]
    r = ty["rule"]
    return {
        "type": ty["name"],
        "type_id": type_id,
        "timecode_sec": t,
        "description": f"ТЕСТОВЫЙ РЕЖИМ (FORTE_FAKE=1): событие сгенерировано заглушкой, а не моделью. {ty['why']}",
        "rule": {"document": r["document"], "short": r.get("short"), "clause": r["clause"], "url": r["url"],
                 "quote": r.get("quote"), "strength": ty.get("strength")},
        "confidence": conf,
        "evidence_frames": frames,
    }


def jpeg_at(path: str, t: float, max_side: int = 960) -> str:
    """Base64 JPEG of the frame at `t` seconds (display rotation applied by ffmpeg)."""
    vf = f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease"
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", "-vf", vf,
         "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
        check=True, capture_output=True).stdout
    if not out:
        raise RuntimeError(f"no frame at {t:.2f} s")
    return base64.b64encode(out).decode("ascii")


class FakeAnalyzer:
    models = ["заглушка FORTE_FAKE (без моделей)"]

    def __init__(self, device: str = "cuda:0"):
        self.delay = float(os.environ.get("FORTE_FAKE_SEC", "4"))

    def analyze(self, video_path: str, workdir: str,
                progress: Callable[[float, str], None] | None = None) -> dict:
        t0 = time.monotonic()
        meta = probe(video_path)
        steps = ["Декодирование видео", "Детекция людей и объектов", "Анализ событий", "Формирование отчёта"]
        for i, stage in enumerate(steps):
            if progress:
                progress(i / len(steps), stage)
            time.sleep(self.delay / len(steps))
        dur = max(meta.duration, 0.1)
        t = round(min(dur * 0.3, max(dur - 0.2, 0.0)), 1)
        t2 = round(min(t + 1.0, max(dur - 0.1, 0.0)), 1)
        t3 = round(min(dur * 0.75, max(dur - 0.1, 0.0)), 1)
        frames = [{"timestamp_sec": ts, "image_base64": jpeg_at(video_path, ts)} for ts in sorted({t, t2})]
        violations = [fake_violation("T1", t, frames, 0.62),
                      fake_violation("T3", t3, [{"timestamp_sec": t3, "image_base64": jpeg_at(video_path, t3)}], 0.48)]
        return {
            "verdict": "violation",
            "violations": violations,
            "meta": {"duration_sec": round(meta.duration, 2), "fps": round(meta.fps, 2), "frames_analyzed": 0,
                     "processing_sec": round(time.monotonic() - t0, 2), "stage_sec": {"fake": self.delay},
                     "models": self.models},
        }
