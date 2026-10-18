"""People in the cash room: detection, tracking, roles and events.

Pipeline per video (frames from `pipeline.video.decode(path, fps=5, max_side=960, pix_fmt="bgr24")`):
1. YOLO26x person boxes (imgsz 640, conf >= 0.25); duplicate boxes on one person are suppressed
   (intersection >= 70 % of the smaller box -> keep the more confident one).
2. IoU + centre-distance tracker (Hungarian assignment); fragments of one person are stitched (a track that
   starts <= 1.5 s after another ended, close to where it ended); a track is split again where one walking
   person vanishes and another walks in at the same spot (visitor handover at the door).
3. Box class: room | outside | reflection. Foot point (box bottom) above the glass line -> outside (conf >= 0.6)
   or reflection (weaker). The line is a fraction of the frame height (340/848, the open-footage camera), a
   camera-specific prior. Below the line a track counts as a room person only after >= 3 confident (>= 0.6)
   frames; weaker tracks are reflections (reflections in the left glass panel reach conf 0.42-0.49 and sometimes
   dip below the line).
4. Roles: cashier = room track with the longest presence (bonus when present at the start), extended across
   leave/return by seat position; other room tracks are visitors; tracks never in the room are unknown.
5. Events: visitor_enter/exit; cashier_stand (seated -> standing transition before a leave: box height jumps
   above 0.72 of the standing height after >= 2 s of sitting); cashier_leave (last frame in the room, with
   t_stand / t_walk_out / t_near_exit helpers); cashier_return; crowd_start/end.
6. Crowd (> 2 people in the room at once, >= 1 s): candidates come from the room count and, line-free, from
   >= 3 confident boxes anywhere (so a new camera angle where the line is wrong still gets flagged). Every
   room-count candidate is verified by the local VLM on 3 frames (people physically in the room, reflections and
   people behind the glass excluded) and survives a VLM outage; line-free candidates are reported in crowd_checks
   (by_line=false) and become alarms only with verify_line_free=True and a VLM yes.

No face recognition or identification: only person boxes and their geometry.
"""
from __future__ import annotations

import os
import re
import threading
import time

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

DEFAULT_WEIGHTS = "/home/shadeform/det_weights/yolo26x.pt"

DET_CONF = 0.25  # boxes kept for tracking
NEW_TRACK_CONF = 0.40  # a new track needs at least this confidence
ROOM_CONF = 0.60  # confident frame for confirming a room track
ROOM_CONFIRM = 3  # confident frames below the glass line needed to call a track a room person
OUTSIDE_CONF = 0.60  # above the glass line: confident -> outside, weak -> reflection
DUP_OVERLAP = 0.70
GLASS_FRAC = 340 / 848  # foot line as a fraction of the frame height (prior for the open-footage camera)
MAX_GAP_S = 1.0  # tracker keeps a lost track this long
STITCH_GAP_S = 1.5
CROWD_MIN_S = 1.0
CROWD_MERGE_S = 1.0
EDGE_S = 0.5  # presence within this time of the clip start/end is "already there"/"still there"
LEAVE_MIN_ABSENT_S = 0.4  # absence until the clip end needed for a leave (clips often end ~1 s after the exit)
ABSENT_MIN_S = 3.0  # shorter cashier absences between two presences are detector misses/occlusions
PRESENT_MIN_S = 1.0  # shorter cashier presence blips (passing the door threshold while entering) are dropped
STAND_MIN_S = 1.0
NEAR_EXIT_H = 0.3  # cashier_leave.t_near_exit: distance from the exit point in standing heights
WALK_STILL_V = 0.08  # cashier_leave.t_walk_out: "still" = centre speed below this many box heights per second
WALK_STILL_S = 1.0  # ... held this long; t_walk_out = end of the last still stretch before the leave
SEATED_MIN_S = 2.0  # a stand needs a seated stretch at least this long before it
HANDOVER_GAP_S = 0.6  # a track gap this long where a walking person vanishes and a walking person appears
HANDOVER_MOVE_BEFORE = 0.5  # centre motion over 1.5 s before the gap, in box heights
HANDOVER_MOVE_AFTER = 0.3  # ... and after it: one person left through the door, another came in
CROWD_ANYWHERE_CONF = 0.60  # line-free crowd candidates: >= 3 boxes this confident, wherever they stand

VLM_SYSTEM = ("You look at a frame from a ceiling CCTV camera in a small bank cash room. One wall of the room is "
              "glass with a glass door; through it you can see an office corridor, and the glass also reflects "
              "people who are inside the room.")
VLM_COUNT_PROMPT = ("Count the people who are physically inside this cash room (on its floor, at the desk or "
                    "standing in the room). Do NOT count people seen through the glass wall in the corridor, and "
                    "do NOT count mirror reflections in the glass. First list each person briefly (where, what "
                    "they do, real/reflection/behind glass), then give the final line 'COUNT: N'.")


def _overlap_of_smaller(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def _dedupe(boxes: list[list[float]]) -> list[list[float]]:
    kept: list[list[float]] = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(_overlap_of_smaller(b, k) < DUP_OVERLAP for k in kept):
            kept.append(b)
    return kept


def _intervals(flags: list[bool], times: list[float], min_s: float, merge_s: float) -> list[tuple[int, int]]:
    """Index runs [i0, i1] where flags hold; gaps of at most merge_s seconds between flagged frames are bridged
    (consecutive frames always join), runs shorter than min_s are dropped."""
    dt = (times[1] - times[0]) if len(times) > 1 else 0.0
    join = max(merge_s, dt) + 1e-6
    runs: list[list[int]] = []
    for i, f in enumerate(flags):
        if not f:
            continue
        if runs and times[i] - times[runs[-1][1]] <= join:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(a, b) for a, b in runs if times[b] - times[a] + dt >= min_s]


class PersonTracker:
    def __init__(self, device: str = "cuda:0", weights: str = DEFAULT_WEIGHTS, imgsz: int = 640, batch: int = 8,
                 glass_frac: float = GLASS_FRAC, vlm_verify: bool = True, verify_line_free: bool = False):
        from ultralytics import YOLO, settings
        settings.update({"sync": False})
        self.device = device
        self.imgsz = imgsz
        self.batch = batch
        self.glass_frac = glass_frac
        self.vlm_verify = vlm_verify
        self.verify_line_free = verify_line_free
        self.model = YOLO(weights)
        self.model_name = os.path.splitext(os.path.basename(weights))[0]
        self._lock = threading.Lock()
        self._vlm = None
        if vlm_verify:
            from pipeline.vlm import VlmClient
            self._vlm = VlmClient(concurrency=8, timeout=60.0)
        # Warm-up so the first video does not pay for CUDA init.
        self._detect([np.zeros((848, 464, 3), np.uint8)])

    # ---------------------------------------------------------------- detection
    def _detect(self, frames_bgr: list[np.ndarray]) -> list[list[list[float]]]:
        out: list[list[list[float]]] = []
        with self._lock:
            for i in range(0, len(frames_bgr), self.batch):
                res = self.model.predict(frames_bgr[i:i + self.batch], imgsz=self.imgsz, conf=DET_CONF,
                                         classes=[0], device=self.device, half=True, verbose=False)
                for r in res:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    conf = r.boxes.conf.cpu().numpy()
                    out.append([[float(v) for v in b] + [float(c)] for b, c in zip(xyxy, conf)])
        return out

    # ---------------------------------------------------------------- tracking
    @staticmethod
    def _track(dets: list[list[list[float]]], times: list[float]) -> list[list[int]]:
        """Track id per detection (-1 = unassigned weak box)."""
        from scipy.optimize import linear_sum_assignment
        tracks: dict[int, dict] = {}  # id -> {box, t}
        next_id = 1
        ids_per_frame: list[list[int]] = []
        for fi, boxes in enumerate(dets):
            t = times[fi]
            live = [tid for tid, tr in tracks.items() if t - tr["t"] <= MAX_GAP_S + 1e-6]
            ids = [-1] * len(boxes)
            if boxes and live:
                a = np.array([tracks[tid]["box"] for tid in live], np.float32)
                b = np.array([bx[:4] for bx in boxes], np.float32)
                iou = _iou_matrix(a, b)
                ca = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], 1)
                cb = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
                dist = np.linalg.norm(ca[:, None] - cb[None], axis=2)
                scale = np.maximum(a[:, 3] - a[:, 1], 1.0)[:, None]
                nd = dist / scale
                cost = (1 - iou) + 0.5 * nd
                gate = (iou >= 0.1) | (nd <= 0.5)
                cost[~gate] = 1e6
                r_idx, c_idx = linear_sum_assignment(cost)
                for r, c in zip(r_idx, c_idx):
                    if cost[r, c] < 1e5:
                        ids[c] = live[r]
            for j, bx in enumerate(boxes):
                if ids[j] == -1 and bx[4] >= NEW_TRACK_CONF:
                    ids[j] = next_id
                    next_id += 1
                if ids[j] != -1:
                    tracks[ids[j]] = {"box": bx[:4], "t": t}
            ids_per_frame.append(ids)
        return ids_per_frame

    # ---------------------------------------------------------------- main entry
    def run(self, frames_bgr: list[np.ndarray], times: list[float], glass_frac: float | None = None) -> dict:
        """glass_frac: foot-line override for this video (the analyzer anchors it on the VLM desk box)."""
        t0 = time.time()
        if not frames_bgr:
            return {"frames": [], "tracks": [], "events": [], "crowd_intervals": [], "cashier_absent": [],
                    "meta": {"n_frames": 0}}
        dets = [_dedupe(b) for b in self._detect(frames_bgr)]
        t_det = time.time() - t0
        gf = self.glass_frac if glass_frac is None else glass_frac
        res = describe(dets, times, frames_bgr[0].shape[0], gf)
        cands = res.pop("_crowd_candidates")
        # VLM checks only the room-count candidates; line-free ones are left unverified (confirmed=None, no
        # alarm) unless verify_line_free is on: the shared VLM is slow under load and states.py already asks it
        # for people_in_room on every sampled frame, so the integrator can fuse that count with these hints.
        to_check = [k for k, c in enumerate(cands) if c[3] or self.verify_line_free]
        got = self._verify_crowd(frames_bgr, [(cands[k][0], cands[k][1]) for k in to_check]) if to_check else []
        verdicts = [{"confirmed": None, "vlm_counts": []} for _ in cands]
        for k, v in zip(to_check, got):
            verdicts[k] = v
        apply_crowd(res, times, cands, verdicts)
        res["meta"].update({"model": self.model_name, "detect_sec": round(t_det, 2), "glass_frac": round(gf, 4),
                            "total_sec": round(time.time() - t0, 2)})
        return res

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _stitch(obs: dict[int, list], times: list[float], line_y: float) -> None:
        """Merge track fragments in place: B starts shortly after A ended, near A's last box."""
        changed = True
        while changed:
            changed = False
            order = sorted(obs, key=lambda k: obs[k][0][0])
            for bid in order:
                b_first_fi, b_first = obs[bid][0]
                best, best_d = None, None
                for aid in order:
                    if aid == bid or aid not in obs:
                        continue
                    a_last_fi, a_last = obs[aid][-1]
                    if a_last_fi >= b_first_fi or times[b_first_fi] - times[a_last_fi] > STITCH_GAP_S:
                        continue
                    # A must not overlap B in time
                    ha = a_last[3] - a_last[1]
                    hb = b_first[3] - b_first[1]
                    ca = ((a_last[0] + a_last[2]) / 2, (a_last[1] + a_last[3]) / 2)
                    cb = ((b_first[0] + b_first[2]) / 2, (b_first[1] + b_first[3]) / 2)
                    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                    if dist > 0.8 * max(ha, hb):
                        continue
                    if (a_last[3] >= line_y) != (b_first[3] >= line_y) and dist > 0.3 * max(ha, hb):
                        continue
                    if best_d is None or dist < best_d:
                        best, best_d = aid, dist
                if best is not None:
                    obs[best].extend(obs.pop(bid))
                    changed = True
                    break

    @staticmethod
    def _cashier(room_frames: dict[int, list[int]], boxes_of: dict, times: list[float]) -> list[int]:
        """Cashier lineage: main track first, then later tracks that take the same seat."""
        if not room_frames:
            return []
        dt = (times[1] - times[0]) if len(times) > 1 else 1.0

        def score(tid):
            fr = room_frames[tid]
            s = len(fr) * dt
            if times[fr[0]] <= EDGE_S + 1e-6:
                s += 10.0
            return s

        main = max(room_frames, key=score)
        lineage = [main]
        seat = PersonTracker._seat_box(room_frames[main], boxes_of[main])
        if seat is None:
            return lineage
        main_frames = set(room_frames[main])
        for tid, fr in room_frames.items():
            if tid == main or set(fr) & main_frames:
                continue
            sb = PersonTracker._seat_box(fr, boxes_of[tid])
            if sb is not None and _iou_matrix(np.array([seat]), np.array([sb]))[0, 0] >= 0.3:
                lineage.append(tid)
        return lineage

    @staticmethod
    def _seat_box(frames: list[int], boxes: dict) -> list[float] | None:
        """Median box over the longest still stretch (the seat), None if the person never stays put."""
        if len(frames) < 5:
            return None
        arr = np.array([boxes[f][:4] for f in frames])
        med = np.median(arr, axis=0)
        iou = _iou_matrix(arr, med[None])[:, 0]
        still = arr[iou >= 0.6]
        if len(still) < 5:
            return None
        return list(np.median(still, axis=0))

    @staticmethod
    def _present_id(rid: set[int], cashier_set: set[int], default: int) -> int:
        inter = rid & cashier_set
        return min(inter) if inter else default

    def _verify_crowd(self, frames_bgr: list[np.ndarray], runs: list[tuple[int, int]]) -> list[dict]:
        """VLM count on up to 3 frames per candidate run; confirmed when most answers give >= 3 people."""
        if self._vlm is None:
            return [{"confirmed": None, "vlm_counts": []} for _ in runs]
        from pipeline.vlm import jpeg_b64
        reqs, owner = [], []
        for k, (a, b) in enumerate(runs):
            picks = sorted({a + (b - a) * q // 4 for q in (1, 2, 3)})
            for i in picks:
                rgb = np.ascontiguousarray(frames_bgr[i][..., ::-1])
                reqs.append({"images": [jpeg_b64(rgb)], "system": VLM_SYSTEM, "prompt": VLM_COUNT_PROMPT,
                             "max_tokens": 220})
                owner.append(k)
        try:
            answers = self._vlm.ask_many(reqs)
        except Exception:  # noqa: BLE001 - VLM down: keep the detector verdict, mark unverified
            answers = [None] * len(reqs)
        counts: list[list[int | None]] = [[] for _ in runs]
        for k, ans in zip(owner, answers):
            m = re.findall(r"COUNT:\s*(\d+)", ans or "")
            counts[k].append(int(m[-1]) if m else None)
        out = []
        for c in counts:
            valid = [x for x in c if x is not None]
            if not valid:
                out.append({"confirmed": None, "vlm_counts": c})
            else:
                out.append({"confirmed": sum(x >= 3 for x in valid) * 2 > len(valid), "vlm_counts": c})
        return out


def _moving(o: list[tuple[int, list[float]]], times: list[float], idx: int, direction: int, win: float) -> float:
    """Centre displacement of a track from o[idx] to its farthest sample within `win` s, in box heights."""
    t0, b0 = times[o[idx][0]], o[idx][1]
    j = idx
    while 0 <= j + direction < len(o) and abs(times[o[j + direction][0]] - t0) <= win + 1e-6:
        j += direction
    b1 = o[j][1]
    dx = (b1[0] + b1[2] - b0[0] - b0[2]) / 2
    dy = (b1[1] + b1[3] - b0[1] - b0[3]) / 2
    return (dx * dx + dy * dy) ** 0.5 / max(b0[3] - b0[1], 1.0)


def _split_handovers(obs: dict[int, list], times: list[float]) -> None:
    """Split a track where one walking person vanishes and another walks in at the same spot (door handover).

    The tracker happily continues a leaving visitor's track with the next visitor coming through the same door.
    """
    next_id = max(obs, default=0) + 1
    for tid in list(obs):
        o = obs[tid]
        cuts = [i + 1 for i in range(len(o) - 1)
                if times[o[i + 1][0]] - times[o[i][0]] >= HANDOVER_GAP_S
                and _moving(o, times, i, -1, 1.5) >= HANDOVER_MOVE_BEFORE
                and _moving(o, times, i + 1, +1, 1.5) >= HANDOVER_MOVE_AFTER]
        if not cuts:
            continue
        bounds = [0, *cuts, len(o)]
        obs[tid] = o[:cuts[0]]
        for s0, s1 in zip(bounds[1:-1], bounds[2:]):
            obs[next_id] = o[s0:s1]
            next_id += 1


def _smooth_presence(present: list[bool], times: list[float]) -> list[bool]:
    """Fill short absences between presences, then drop short presence blips."""
    n = len(present)
    out = list(present)
    dt = (times[1] - times[0]) if n > 1 else 0.0
    runs = _intervals(out, times, 0.0, 0.0)
    for (a0, b0), (a1, _) in zip(runs, runs[1:]):
        if (a1 - b0 - 1) * dt < ABSENT_MIN_S:
            out[b0 + 1:a1] = [True] * (a1 - b0 - 1)
    for a, b in _intervals(out, times, 0.0, 0.0):
        if (b - a + 1) * dt < PRESENT_MIN_S:
            out[a:b + 1] = [False] * (b - a + 1)
    return out


def _stand_time(boxes: dict[int, list[float]], times: list[float], fa: int, fb: int) -> float | None:
    """Moment the cashier gets up from the seat before leaving at frame fb (presence run starts at fa).

    Seated = box height below 0.72 of the person's standing height (90th percentile of the run) and not tall-narrow.
    Returns the first frame of the last seated -> standing transition that follows >= SEATED_MIN_S of sitting and
    is held >= STAND_MIN_S (or until the leave). None when the person never sat in this run.
    """
    fis = sorted(f for f in boxes if fa <= f <= fb)
    if len(fis) < 5:
        return None
    arr = np.array([boxes[f][:4] for f in fis])
    h = arr[:, 3] - arr[:, 1]
    w = arr[:, 2] - arr[:, 0]
    h_stand = float(np.percentile(h, 90))
    seated = (h < 0.72 * h_stand) & (h / np.maximum(w, 1.0) < 2.0)
    stand_t = None
    i = 0
    while i < len(fis):
        if seated[i]:
            j = i
            while j + 1 < len(fis) and seated[j + 1]:
                j += 1
            if times[fis[j]] - times[fis[i]] >= SEATED_MIN_S and j + 1 < len(fis):
                k = j + 1
                while k + 1 < len(fis) and not seated[k + 1]:
                    k += 1
                if times[fis[k]] - times[fis[j + 1]] >= STAND_MIN_S or k == len(fis) - 1:
                    stand_t = times[fis[j + 1]]
            i = j + 1
        else:
            i += 1
    return stand_t


def _near_exit_time(boxes: dict[int, list[float]], times: list[float], fa: int, fb: int) -> float | None:
    """Last moment the cashier was still >= NEAR_EXIT_H standing heights away from where she left the frame.

    The exit point is the mean foot point of the last 3 boxes, so no door position is assumed.
    """
    fis = sorted(f for f in boxes if fa <= f <= fb)
    if len(fis) < 5:
        return None
    arr = np.array([boxes[f][:4] for f in fis])
    cx = (arr[:, 0] + arr[:, 2]) / 2
    fy = arr[:, 3]
    h = float(np.percentile(arr[:, 3] - arr[:, 1], 90))
    d = np.hypot(cx - cx[-3:].mean(), fy - fy[-3:].mean()) / max(h, 1.0)
    far = np.nonzero(d >= NEAR_EXIT_H)[0]
    return times[fis[far[-1]]] if len(far) else None


def _walk_out_time(boxes: dict[int, list[float]], times: list[float], fa: int, fb: int) -> float | None:
    """End of the last still stretch (>= WALK_STILL_S) before the leave: the start of the final walk out."""
    fis = sorted(f for f in boxes if fa <= f <= fb)
    if len(fis) < 5:
        return None
    ts = np.array([times[f] for f in fis])
    arr = np.array([boxes[f][:4] for f in fis])
    cx = (arr[:, 0] + arr[:, 2]) / 2
    cy = (arr[:, 1] + arr[:, 3]) / 2
    h = max(float(np.median(arr[:, 3] - arr[:, 1])), 1.0)
    j0 = np.searchsorted(ts, ts - 1.0)
    j1 = np.searchsorted(ts, ts + 1.0, side="right") - 1
    v = np.hypot(cx[j1] - cx[j0], cy[j1] - cy[j0]) / np.maximum(ts[j1] - ts[j0], 1e-3) / h
    still = v < WALK_STILL_V
    i = len(ts) - 1
    while i >= 0:
        if still[i]:
            j = i
            while j > 0 and still[j - 1]:
                j -= 1
            if ts[i] - ts[j] >= WALK_STILL_S:
                return float(ts[i])
            i = j - 1
        else:
            i -= 1
    return None


def describe(dets: list[list[list[float]]], times: list[float], frame_h: int, glass_frac: float = GLASS_FRAC) -> dict:
    """Tracks, roles, events and crowd candidates from per-frame deduplicated boxes [x1, y1, x2, y2, conf].

    Pure function of the detections (no model, no VLM), so the rules can be re-tuned on cached boxes.
    """
    line_y = glass_frac * frame_h
    ids = PersonTracker._track(dets, times)
    n = len(times)
    dur = times[-1] + ((times[1] - times[0]) if n > 1 else 0.0)

    # Per-track observations: list of (frame_idx, box)
    obs: dict[int, list[tuple[int, list[float]]]] = {}
    for fi, (boxes, tids) in enumerate(zip(dets, ids)):
        for b, tid in zip(boxes, tids):
            if tid != -1:
                obs.setdefault(tid, []).append((fi, b))
    PersonTracker._stitch(obs, times, line_y)
    _split_handovers(obs, times)

    # Track-level room confirmation.
    room_track = {tid: sum(1 for _, b in o if b[3] >= line_y and b[4] >= ROOM_CONF) >= ROOM_CONFIRM
                  for tid, o in obs.items()}

    def box_cls(tid: int, b: list[float]) -> str:
        if b[3] < line_y:
            return "outside" if b[4] >= OUTSIDE_CONF else "reflection"
        return "room" if tid != -1 and room_track.get(tid) else "reflection"

    frames_out = []
    room_ids_per_frame: list[set[int]] = []
    tid_of = {}
    for tid, o in obs.items():
        for fi, b in o:
            tid_of[(fi, id(b))] = tid
    for fi, boxes in enumerate(dets):
        row, rid = [], set()
        for b in boxes:
            tid = tid_of.get((fi, id(b)), -1)
            c = box_cls(tid, b)
            if c == "room":
                rid.add(tid)
            row.append([round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1), round(b[4], 3), tid, c])
        frames_out.append({"t": times[fi], "n_room": len(rid), "boxes": row})
        room_ids_per_frame.append(rid)

    # Room presence per track (frame indices).
    room_frames: dict[int, list[int]] = {}
    for fi, rid in enumerate(room_ids_per_frame):
        for tid in rid:
            room_frames.setdefault(tid, []).append(fi)

    boxes_of = {tid: {fi: b for fi, b in o} for tid, o in obs.items()}
    cashier_ids = PersonTracker._cashier(room_frames, boxes_of, times)
    cashier_set = set(cashier_ids)

    events: list[dict] = []
    dt = (times[1] - times[0]) if n > 1 else 0.0
    cid = cashier_ids[0] if cashier_ids else -1
    cashier_boxes: dict[int, list[float]] = {}
    for tid in cashier_ids:
        for fi, b in boxes_of[tid].items():
            if tid in room_ids_per_frame[fi]:
                cashier_boxes.setdefault(fi, b)
    present = _smooth_presence([bool(room_ids_per_frame[fi] & cashier_set) for fi in range(n)], times)
    runs = _intervals(present, times, min_s=0.0, merge_s=0.0)  # presence runs, already cleaned
    cashier_absent = []
    if runs:
        a0 = runs[0][0]
        if a0 > 0:  # cashier walks in after the clip starts
            cashier_absent.append([0.0, round(times[a0], 2)])
            events.append({"type": "cashier_return", "t": round(times[a0], 2),
                           "track_id": PersonTracker._present_id(room_ids_per_frame[a0], cashier_set, cid)})
        for k, (a, b) in enumerate(runs):
            last = k == len(runs) - 1
            if last and (n - 1 - b) * dt < LEAVE_MIN_ABSENT_S:
                continue  # still in the room at the end of the clip
            tid_leave = PersonTracker._present_id(room_ids_per_frame[b], cashier_set, cid)
            stand = _stand_time(cashier_boxes, times, a, b)
            if stand is not None:
                events.append({"type": "cashier_stand", "t": round(stand, 2), "track_id": tid_leave})
            near = _near_exit_time(cashier_boxes, times, a, b)
            walk = _walk_out_time(cashier_boxes, times, a, b)
            events.append({"type": "cashier_leave", "t": round(times[b], 2), "track_id": tid_leave,
                           "t_stand": None if stand is None else round(stand, 2),
                           "t_walk_out": None if walk is None else round(walk, 2),
                           "t_near_exit": None if near is None else round(near, 2)})
            if last:
                cashier_absent.append([round(times[b], 2), round(dur, 2)])
            else:
                a2 = runs[k + 1][0]
                cashier_absent.append([round(times[b], 2), round(times[a2], 2)])
                events.append({"type": "cashier_return", "t": round(times[a2], 2),
                               "track_id": PersonTracker._present_id(room_ids_per_frame[a2], cashier_set, cid)})

    # Visitors.
    tracks_out = []
    for tid, o in sorted(obs.items(), key=lambda kv: kv[1][0][0]):
        rf = room_frames.get(tid)
        if tid in cashier_set:
            role = "cashier"
        elif rf:
            role = "visitor"
        else:
            role = "unknown"
        cls_counts: dict[str, int] = {}
        for fi, b in o:
            c = box_cls(tid, b)
            cls_counts[c] = cls_counts.get(c, 0) + 1
        tr = {"id": tid, "t_start": round(times[o[0][0]], 2), "t_end": round(times[o[-1][0]], 2), "role": role,
              "cls": max(cls_counts, key=cls_counts.get), "n_frames": len(o)}
        if rf:
            tr["t_room_start"] = round(times[rf[0]], 2)
            tr["t_room_end"] = round(times[rf[-1]], 2)
        tracks_out.append(tr)
        if role == "visitor" and (rf[-1] - rf[0] + 1) * dt >= PRESENT_MIN_S:
            # At the clip start a visitor still walking in (moving >= 0.3 box heights in the first 2 s) also enters.
            if times[rf[0]] > EDGE_S or _moving(o, times, 0, +1, 2.0) >= 0.3:
                events.append({"type": "visitor_enter", "t": round(times[rf[0]], 2), "track_id": tid})
            if dur - times[rf[-1]] > EDGE_S + dt:
                events.append({"type": "visitor_exit", "t": round(times[rf[-1]], 2), "track_id": tid})

    # Crowd candidates: > 2 people in the room at once; confirmed later by apply_crowd.
    # Two sources: (a) > 2 room-class people; (b) line-free: >= 3 confident boxes anywhere with >= 2 of them in
    # the room, so a third person whose feet fall above a misplaced glass line on a new camera angle is still
    # checked. (b)-only candidates are kept only when the VLM confirms them.
    by_line = [len(r) > 2 for r in room_ids_per_frame]
    anywhere = [len(r) >= 2 and sum(1 for b in boxes if b[4] >= CROWD_ANYWHERE_CONF) >= 3
                for r, boxes in zip(room_ids_per_frame, dets)]
    line_runs = _intervals(by_line, times, CROWD_MIN_S, CROWD_MERGE_S)
    free_runs = [(a, b) for a, b in _intervals(anywhere, times, CROWD_MIN_S, CROWD_MERGE_S)
                 if not any(a <= b2 and a2 <= b for a2, b2 in line_runs)]
    cands = sorted([(a, b, max(len(room_ids_per_frame[i]) for i in range(a, b + 1)), True) for a, b in line_runs]
                   + [(a, b, max(len(room_ids_per_frame[i]) for i in range(a, b + 1)), False) for a, b in free_runs])

    events.sort(key=lambda e: e["t"])
    return {
        "frames": frames_out,
        "tracks": tracks_out,
        "events": events,
        "crowd_intervals": [],
        "cashier_absent": cashier_absent,
        "crowd_checks": [],
        "meta": {"n_frames": n, "duration_sec": round(dur, 2), "glass_line_y": round(line_y, 1)},
        "_crowd_candidates": cands,
    }


def apply_crowd(res: dict, times: list[float], cands: list[tuple[int, int, int, bool]],
                verdicts: list[dict]) -> None:
    """Keep confirmed crowd candidates; add crowd_start/crowd_end events.

    VLM yes -> kept; VLM no -> dropped; VLM unavailable -> kept only if the detector itself counted > 2 in the room.
    """
    if not verdicts:
        verdicts = [{"confirmed": None, "vlm_counts": []} for _ in cands]
    for (a, b, max_n, by_line), ver in zip(cands, verdicts):
        res["crowd_checks"].append({"t0": times[a], "t1": times[b], "max_n": max_n, "by_line": by_line, **ver})
        if ver["confirmed"] is False or (ver["confirmed"] is None and not by_line):
            continue
        res["crowd_intervals"].append([times[a], times[b], max(max_n, 3)])
        res["events"].append({"type": "crowd_start", "t": times[a], "track_id": -1})
        res["events"].append({"type": "crowd_end", "t": times[b], "track_id": -1})
    res["events"].sort(key=lambda e: e["t"])
