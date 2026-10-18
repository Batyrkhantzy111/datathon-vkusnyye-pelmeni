"""Simulated new-camera augmentations of open clips for the robustness stress test (spec FAQ 4, criterion 5).

Writes OUT/X-NN__<aug>.mov (H.264, yuv420p, 30 fps, no audio) for every clip x augmentation:
  persp  - perspective warp emulating a camera moved ~15-20 deg sideways (right side nearer, left farther)
  rot    - 8 deg rotation, zoomed just enough to crop the black corners away
  zoom   - centre crop to 80 % and scale back (camera closer)
  flip   - horizontal mirror (mirrored room layout)
  dark   - lower brightness and contrast + slight Gaussian blur
  land   - landscape 16:9: the portrait frame padded to 16:9 (a landscape camera with the scene in the middle)

Usage (GPU host, plain python3, needs ffmpeg/ffprobe):
  python3 analysis/robust/make_aug.py --videos ~/forte/videos --out ~/forte/robust [--clips X-18,X-15] [--augs persp,rot]
Then: ~/det-venv/bin/python -m pipeline.batch --videos ~/forte/robust --out ~/forte/cache/robust --threads 2
and score with analysis/robust/score.py (maps X-NN__aug back to X-NN labels).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

CLIPS = ["X-18", "X-15", "X-28", "X-20", "X-30", "X-02", "X-04", "X-01", "X-05", "X-23", "X-29"]
ROT_DEG = 8.0


def _size(path: pathlib.Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])


def _even(x: float) -> int:
    return int(round(x / 2)) * 2


def filters(w: int, h: int) -> dict[str, str]:
    a = math.radians(ROT_DEG)
    # scale k so the rotated w x h frame still covers the whole w x h output (no black corners)
    k = max(math.cos(a) + (h / w) * math.sin(a), math.cos(a) + (w / h) * math.sin(a))
    lw = _even(h * 16 / 9) if w < h * 16 / 9 else w
    lh = h if w < h * 16 / 9 else _even(w * 9 / 16)
    # perspective (sense=source): the source quad mapped onto the full output. The right edge of the source quad
    # is 70 % of the height and the left edge is taken from outside the frame -> right side magnified, left side
    # shrunk with a black wedge, as when the camera yaws ~15-20 deg.
    dy_r, dx_l = 0.15 * h, 0.06 * w
    persp = (f"perspective=x0={-dx_l:.0f}:y0={-0.04 * h:.0f}:x1={w}:y1={dy_r:.0f}:"
             f"x2={-dx_l:.0f}:y2={1.04 * h:.0f}:x3={w}:y3={h - dy_r:.0f}:interpolation=linear")
    return {
        "persp": persp,
        "rot": f"scale={_even(w * k)}:{_even(h * k)},rotate={a:.5f}:ow={w}:oh={h}:bilinear=1",
        "zoom": f"crop={_even(w * 0.8)}:{_even(h * 0.8)},scale={w}:{h}",
        "flip": "hflip",
        "dark": "eq=brightness=-0.18:contrast=0.7:saturation=0.85,gblur=sigma=1.5",
        "land": f"pad={lw}:{lh}:(ow-iw)/2:(oh-ih)/2:black",
    }


def make(src: pathlib.Path, dst: pathlib.Path, vf: str) -> str:
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", f"{vf},format=yuv420p", "-r", "30",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
    return dst.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(pathlib.Path.home() / "forte/videos"))
    ap.add_argument("--out", default=str(pathlib.Path.home() / "forte/robust"))
    ap.add_argument("--clips", default=",".join(CLIPS))
    ap.add_argument("--augs", default="persp,rot,zoom,flip,dark,land")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for clip in a.clips.split(","):
        src = pathlib.Path(a.videos) / f"{clip}.mov"
        fl = filters(*_size(src))
        for aug in a.augs.split(","):
            jobs.append((src, out / f"{clip}__{aug}.mov", fl[aug]))
    with ThreadPoolExecutor(max_workers=6) as ex:
        for name in ex.map(lambda j: make(*j), jobs):
            print(name, flush=True)


if __name__ == "__main__":
    main()
