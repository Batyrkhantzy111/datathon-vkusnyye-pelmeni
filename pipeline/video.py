"""Video decoding shared by every pipeline stage.

ffmpeg applies the display-matrix rotation (autorotate is on by default), then the fps and scale filters.
Frame k of a decode at `fps` sits at k / fps seconds from the start of the video.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    duration: float  # seconds
    fps: float  # native frame rate
    width: int  # display size after rotation, before scaling
    height: int
    rotation: int  # degrees, from the display matrix or the legacy rotate tag


def probe(path: str) -> VideoMeta:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", path],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    if not info.get("streams"):
        raise ValueError(f"no video stream in {path}")
    st = info["streams"][0]
    rot = 0
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if not rot and "rotate" in (st.get("tags") or {}):
        rot = int(st["tags"]["rotate"])
    w, h = int(st["width"]), int(st["height"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    num, den = (st.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    dur = float(st.get("duration") or (info.get("format") or {}).get("duration") or 0.0)
    return VideoMeta(duration=dur, fps=fps, width=w, height=h, rotation=rot)


def target_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Even output size with the long side at most `max_side`, aspect ratio kept."""
    s = min(1.0, max_side / max(width, height))
    return max(2, int(round(width * s / 2)) * 2), max(2, int(round(height * s / 2)) * 2)


def decode(path: str, fps: float, max_side: int = 960, pix_fmt: str = "rgb24"
           ) -> tuple[list[np.ndarray], list[float], VideoMeta]:
    """Frames sampled at `fps` and their timestamps in seconds.

    pix_fmt: "rgb24" for PIL / the VLM, "bgr24" for Ultralytics and OpenCV.
    The frames are read-only views into one buffer: copy a frame before drawing on it.
    """
    if pix_fmt not in ("rgb24", "bgr24"):
        raise ValueError(pix_fmt)
    meta = probe(path)
    tw, th = target_size(meta.width, meta.height, max_side)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-an", "-sn", "-dn",
           "-vf", f"fps={fps},scale={tw}:{th}:flags=area", "-pix_fmt", pix_fmt, "-f", "rawvideo", "pipe:1"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    size = tw * th * 3
    n = len(raw) // size
    arr = np.frombuffer(raw, np.uint8, count=n * size).reshape(n, th, tw, 3)
    return list(arr), [round(k / fps, 3) for k in range(n)], meta


def frames_at(path: str, times: list[float], max_side: int = 960) -> list[np.ndarray]:
    """RGB frames at the given timestamps (seconds), one fast input-seek ffmpeg call per timestamp."""
    meta = probe(path)
    tw, th = target_size(meta.width, meta.height, max_side)
    out = []
    for t in times:
        t = min(max(0.0, float(t)), max(0.0, meta.duration - 0.05))
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", path, "-an", "-frames:v", "1",
               "-vf", f"scale={tw}:{th}:flags=area", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout
        if len(raw) < tw * th * 3:  # seek past the last decodable frame: fall back to the final frame
            cmd[4:6] = ["-sseof", "-0.5"]
            raw = subprocess.run(cmd, check=True, capture_output=True).stdout
        out.append(np.frombuffer(raw, np.uint8, count=tw * th * 3).reshape(th, tw, 3))
    return out
