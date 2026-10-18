"""FastAPI service: video upload, job queue over a shared Analyzer, results API (spec §10) and the web UI.

Run from ~/forte:  python -m uvicorn service.app:app --host 127.0.0.1 --port 8000
Env:
  FORTE_API_KEY       API key; else read from FORTE_API_KEY_FILE (default ~/forte/service/api_key.txt),
                      generated there on first start if missing.
  FORTE_JOBS_DIR      job storage (default ~/forte/jobs)
  FORTE_WORKERS       worker threads sharing one Analyzer (default 2)
  FORTE_FAKE=1        use service.fake.FakeAnalyzer instead of pipeline.analyze.Analyzer
  FORTE_DEVICE        device passed to Analyzer (default cuda:0)
  FORTE_MAX_UPLOAD_MB upload limit (default 200)
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import queue
import secrets
import shutil
import subprocess
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile

from pipeline.video import probe

log = logging.getLogger("forte.service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

HOME = Path.home()
JOBS_DIR = Path(os.environ.get("FORTE_JOBS_DIR", HOME / "forte" / "jobs"))
KEY_FILE = Path(os.environ.get("FORTE_API_KEY_FILE", HOME / "forte" / "service" / "api_key.txt"))
STATIC_DIR = Path(__file__).resolve().parent / "static"
N_WORKERS = max(1, int(os.environ.get("FORTE_WORKERS", "2")))
FAKE = os.environ.get("FORTE_FAKE", "") == "1"
DEVICE = os.environ.get("FORTE_DEVICE", "cuda:0")
MAX_UPLOAD = int(float(os.environ.get("FORTE_MAX_UPLOAD_MB", "200")) * 1024 * 1024)
VIDEO_EXT = {".mov": "video/mp4", ".mp4": "video/mp4", ".m4v": "video/mp4", ".qt": "video/mp4"}
CHUNK = 1 << 20


def load_api_key() -> str:
    key = os.environ.get("FORTE_API_KEY", "").strip()
    if key:
        return key
    if KEY_FILE.exists() and KEY_FILE.read_text().strip():
        return KEY_FILE.read_text().strip()
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(32)
    KEY_FILE.write_text(key + "\n")
    KEY_FILE.chmod(0o600)
    log.info("generated a new API key in %s", KEY_FILE)
    return key


API_KEY = load_api_key()


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


class Jobs:
    """Job states in memory, mirrored to <JOBS_DIR>/<id>/status.json; results live in result.json."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.status: dict[str, dict] = {}
        self.queue: queue.Queue[str] = queue.Queue()
        self.analyzer = None
        self.analyzer_state = "loading"  # loading | ready | error
        self.analyzer_error = ""
        self.analyzer_ready = threading.Event()
        self.busy = 0

    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def new_id(self) -> str:
        while True:
            job_id = secrets.token_hex(6)
            if not self.dir(job_id).exists():
                return job_id

    def save(self, job_id: str) -> None:
        with self.lock:
            st = dict(self.status[job_id])
        write_json(self.dir(job_id) / "status.json", st)

    def update(self, job_id: str, **fields) -> None:
        with self.lock:
            self.status[job_id].update(fields)
        self.save(job_id)

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            st = self.status.get(job_id)
            return dict(st) if st else None

    def add(self, job_id: str, video: Path, filename: str, meta) -> None:
        st = {"job_id": job_id, "status": "processing", "progress": 0.0, "stage": "В очереди",
              "filename": filename, "video": video.name, "created_at": time.time(),
              "video_duration_sec": round(meta.duration, 2)}
        with self.lock:
            self.status[job_id] = st
        self.save(job_id)
        self.queue.put(job_id)

    def restore(self) -> None:
        """Load finished jobs; re-queue those left in processing by a restart, oldest first."""
        pending = []
        for d in self.root.iterdir():
            f = d / "status.json"
            if not f.is_file():
                continue
            try:
                st = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            job_id = st.get("job_id") or d.name
            self.status[job_id] = st
            if st.get("status") == "processing":
                st.update(progress=0.0, stage="В очереди (перезапуск сервиса)")
                st.pop("started_at", None)
                pending.append((st.get("created_at", 0.0), job_id))
        for _, job_id in sorted(pending):
            self.save(job_id)
            self.queue.put(job_id)
        log.info("restored %d jobs, re-queued %d", len(self.status), len(pending))

    def load_analyzer(self) -> None:
        try:
            if FAKE:
                from service.fake import FakeAnalyzer
                self.analyzer = FakeAnalyzer(DEVICE)
            else:
                from pipeline.analyze import Analyzer
                self.analyzer = Analyzer(DEVICE)
            self.analyzer_state = "ready"
            log.info("analyzer ready: %s", type(self.analyzer).__name__)
        except Exception as e:  # keep the API up and fail jobs with the reason
            self.analyzer_state = "error"
            self.analyzer_error = f"{type(e).__name__}: {e}"
            log.exception("analyzer failed to load")
        finally:
            self.analyzer_ready.set()

    def worker(self) -> None:
        while True:
            job_id = self.queue.get()
            try:
                self.run(job_id)
            except Exception:
                log.exception("worker error on %s", job_id)
            finally:
                self.queue.task_done()

    def run(self, job_id: str) -> None:
        st = self.get(job_id)
        if not st or st.get("status") != "processing":
            return
        with self.lock:
            self.busy += 1
        t0 = time.time()
        self.update(job_id, started_at=t0, stage="Загрузка моделей" if not self.analyzer_ready.is_set()
                    else "Анализ", progress=0.0)
        try:
            self.analyzer_ready.wait()
            if self.analyzer is None:
                raise RuntimeError(f"анализатор не загрузился: {self.analyzer_error}")
            d = self.dir(job_id)
            work = d / "work"
            work.mkdir(exist_ok=True)

            def progress(p: float, stage: str) -> None:
                with self.lock:
                    self.status[job_id].update(progress=round(min(max(float(p), 0.0), 0.99), 3),
                                               stage=str(stage))
                self.save(job_id)

            result = self.analyzer.analyze(str(d / st["video"]), str(work), progress)
            dt = round(time.time() - t0, 2)
            violations = result.get("violations") or []
            result["verdict"] = "violation" if violations else "no_violation"
            result["violations"] = violations
            result.setdefault("meta", {})["job_processing_sec"] = dt
            result["meta"].setdefault("processing_sec", dt)
            write_json(d / "result.json", result)
            self.update(job_id, status="done", progress=1.0, stage="Готово", processing_sec=dt,
                        finished_at=time.time())
            shutil.rmtree(work, ignore_errors=True)
            log.info("job %s done in %.1f s: %s, %d violations", job_id, dt, result["verdict"], len(violations))
        except Exception as e:
            dt = round(time.time() - t0, 2)
            log.error("job %s failed after %.1f s\n%s", job_id, dt, traceback.format_exc())
            self.update(job_id, status="failed", stage="Ошибка", error=f"{type(e).__name__}: {e}",
                        processing_sec=dt, finished_at=time.time())
        finally:
            with self.lock:
                self.busy -= 1

    def response(self, job_id: str) -> dict:
        st = self.get(job_id)
        if st is None:
            raise HTTPException(404, "задача не найдена")
        out = {"job_id": job_id, "status": st["status"]}
        if st["status"] == "processing":
            out.update(progress=st.get("progress", 0.0), stage=st.get("stage", ""),
                       queue_position=self.position(job_id))
        elif st["status"] == "failed":
            out["error"] = st.get("error", "неизвестная ошибка")
        else:
            result = json.loads((self.dir(job_id) / "result.json").read_text())
            out.update(verdict=result.get("verdict"), violations=result.get("violations", []))
            out.update({k: v for k, v in result.items() if k not in out})
        for k in ("filename", "created_at", "processing_sec", "video_duration_sec"):
            if k in st:
                out[k] = st[k]
        return out

    def position(self, job_id: str) -> int:
        with self.queue.mutex:
            q = list(self.queue.queue)
        return q.index(job_id) + 1 if job_id in q else 0


jobs = Jobs(JOBS_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    jobs.restore()
    threading.Thread(target=jobs.load_analyzer, name="analyzer-load", daemon=True).start()
    for i in range(N_WORKERS):
        threading.Thread(target=jobs.worker, name=f"worker-{i}", daemon=True).start()
    log.info("service up: %d workers, analyzer=%s, jobs in %s", N_WORKERS, "fake" if FAKE else "real", JOBS_DIR)
    yield


app = FastAPI(title="ForteBank cash-discipline video analyzer", version="1.0", lifespan=lifespan)


def check_key(request: Request) -> None:
    key = request.headers.get("x-api-key", "")
    if not key or not hmac.compare_digest(key.encode(), API_KEY.encode()):
        raise HTTPException(401, "неверный или отсутствующий X-API-Key")


def save_upload(upload: UploadFile, dest: Path) -> int:
    size = 0
    upload.file.seek(0)
    with open(dest, "wb") as f:
        while chunk := upload.file.read(CHUNK):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, f"файл больше {MAX_UPLOAD // (1024 * 1024)} МБ")
            f.write(chunk)
    return size


def decode_check(video: Path, duration: float) -> None:
    """Decode one frame at the start and one in the middle; raise if no frame comes out (corrupt media data).
    Concealable decoder warnings are tolerated: rejecting a playable clip costs more than a later failed job."""
    for t in sorted({0.0, round(duration / 2, 2)}):
        try:
            p = subprocess.run(
                ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.2f}", "-i", str(video), "-map", "0:v:0",
                 "-frames:v", "1", "-vf", "scale=64:-2", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"],
                capture_output=True, timeout=5)
        except subprocess.TimeoutExpired:
            raise ValueError(f"декодирование кадра на {t:.1f} с не уложилось в 5 с")
        if p.returncode != 0 or not p.stdout:
            err = [ln.strip() for ln in p.stderr.decode("utf-8", "replace").splitlines()
                   if ln.strip() and "Last message repeated" not in ln]
            raise ValueError(f"не удалось декодировать кадр на {t:.1f} с"
                             + (f" ({err[-1][:160]})" if err else ""))


async def submit(request: Request) -> JSONResponse:
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_UPLOAD + CHUNK:
        raise HTTPException(413, f"файл больше {MAX_UPLOAD // (1024 * 1024)} МБ")
    try:
        form = await request.form(max_files=1, max_fields=10)
    except Exception as e:
        raise HTTPException(400, f"ожидается multipart/form-data с полем file: {e}")
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(400, "нет файла: передайте видео в поле file (multipart/form-data)")
    filename = Path(upload.filename or "video.mov").name
    ext = Path(filename).suffix.lower() or ".mov"
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"неподдерживаемый формат {ext}: принимаются .mov и .mp4")
    job_id = jobs.new_id()
    d = jobs.dir(job_id)
    d.mkdir(parents=True)
    video = d / f"input{ext}"
    try:
        size = await run_in_threadpool(save_upload, upload, video)
        if size == 0:
            raise HTTPException(400, "пустой файл")
        try:
            meta = await run_in_threadpool(probe, str(video))
            if meta.width <= 0 or meta.height <= 0 or meta.duration <= 0:
                raise ValueError("нет видеопотока")
            await run_in_threadpool(decode_check, video, meta.duration)
        except Exception as e:
            reason = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
            if isinstance(reason, bytes):
                reason = reason.decode("utf-8", "replace")
            reason = str(reason or "").replace(str(video), filename).strip().splitlines()
            raise HTTPException(400, "файл не является читаемым видео (проверка ffprobe/ffmpeg)"
                                + (f": {reason[-1][:200]}" if reason else ""))
    except HTTPException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    finally:
        await upload.close()
    jobs.add(job_id, video, filename, meta)
    log.info("job %s queued: %s, %.1f MB, %.1f s", job_id, filename, size / 1e6, meta.duration)
    return JSONResponse({"job_id": job_id, "status": "processing"})


@app.post("/api/v1/analyze")
async def api_analyze(request: Request):
    check_key(request)
    return await submit(request)


@app.get("/api/v1/jobs/{job_id}")
async def api_job(job_id: str, request: Request):
    check_key(request)
    return await run_in_threadpool(jobs.response, job_id)


@app.get("/api/v1/health")
async def health():
    return {"ok": True, "queue": jobs.queue.qsize(), "workers": N_WORKERS, "busy": jobs.busy,
            "analyzer": "fake" if FAKE else "real", "analyzer_state": jobs.analyzer_state}


NORMS_FILE = Path(__file__).resolve().parent.parent / "pipeline" / "norms.json"


def violation_types() -> dict:
    """Public view of the violation-type registry: name, summary, strength and norms per type."""
    reg = json.loads(NORMS_FILE.read_text(encoding="utf-8"))
    keep = ("document", "short", "clause", "url", "quote")
    types = [{"type_id": tid, "name": t["name"], "summary": t.get("summary", ""), "strength": t.get("strength"),
              "rule": {k: t["rule"].get(k) for k in keep},
              "extra_rules": [{k: r.get(k) for k in keep} for r in t.get("extra_rules", [])]}
             for tid, t in reg["types"].items()]
    sources = [{"id": sid, "title": s.get("title"), "requisites": s.get("requisites"), "url": s.get("url"),
                "status": s.get("status")} for sid, s in reg.get("sources", {}).items()]
    return {"version": reg.get("version"), "types": types, "sources": sources}


@app.get("/api/v1/types")
@app.get("/ui/types")
async def types():
    return await run_in_threadpool(violation_types)


@app.post("/ui/analyze")
async def ui_analyze(request: Request):
    return await submit(request)


@app.get("/ui/jobs/{job_id}")
async def ui_job(job_id: str):
    return await run_in_threadpool(jobs.response, job_id)


@app.get("/ui/jobs/{job_id}/video")
async def ui_video(job_id: str):
    st = jobs.get(job_id)
    if st is None:
        raise HTTPException(404, "задача не найдена")
    path = jobs.dir(job_id) / st["video"]
    if not path.is_file():
        raise HTTPException(404, "видео не найдено")
    return FileResponse(path, media_type=VIDEO_EXT.get(path.suffix.lower(), "video/mp4"))


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
