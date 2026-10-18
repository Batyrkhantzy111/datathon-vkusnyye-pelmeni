"""Run the full Analyzer on open clips (as the service does) and score the results.

Usage (GPU host, from ~/forte):
  ~/det-venv/bin/python -m pipeline.batch [--clips X-01,X-02|all] [--threads 2] [--out ~/forte/cache/results]
Writes X-NN.json results (evidence images replaced by their sizes) and X-NN.signals.json, then prints the eval.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from pipeline import eval as ev
from pipeline.analyze import Analyzer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="all")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--videos", default=str(pathlib.Path.home() / "forte/videos"))
    ap.add_argument("--out", default=str(pathlib.Path.home() / "forte/cache/results"))
    ap.add_argument("--gt", default=str(pathlib.Path.home() / "forte/analysis/gt_team.json"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = (sorted(p.stem for p in pathlib.Path(a.videos).glob("X-*.mov")) if a.clips == "all"
             else a.clips.split(","))
    an = Analyzer()

    def one(clip: str):
        job = pathlib.Path(tempfile.mkdtemp(prefix=f"batch_{clip}_"))
        work = job / "work"
        work.mkdir()
        t0 = time.time()
        res = an.analyze(f"{a.videos}/{clip}.mov", str(work))
        dt = time.time() - t0
        for v in res["violations"]:
            for f in v["evidence_frames"]:
                f["image_base64"] = f"<{len(f['image_base64'])} chars>"
        (out / f"{clip}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        sig = job / "signals.json"
        if sig.exists():
            shutil.copy(sig, out / f"{clip}.signals.json")
        shutil.rmtree(job, ignore_errors=True)
        kinds = ",".join(f"{v['type_id']}@{v['timecode_sec']}" for v in res["violations"]) or "-"
        print(f"{clip} {res['meta']['duration_sec']:.0f}s video -> {dt:.1f}s {res['meta']['stage_sec']} {kinds}",
              flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        list(ex.map(one, clips))
    print(f"total {time.time() - t0:.0f}s for {len(clips)} clips", flush=True)
    gt = json.loads(pathlib.Path(a.gt).read_text(encoding="utf-8"))
    preds = {c: json.loads((out / f"{c}.json").read_text(encoding="utf-8")) for c in clips}
    ev.print_report(ev.score(preds, gt))


if __name__ == "__main__":
    main()
