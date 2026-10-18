"""Rules engine: per-video signals -> violation candidates.

Signals (JSON-serializable, so rules can be tuned offline on cached outputs, see pipeline/offline.py):
- states:  per-frame VLM scene states at 1 fps (pipeline/states.py): {t, ok, people_in_room, cashier, safe, laptop,
           cash_on_desk, bag_on_desk, notebook_on_desk, clothing_on_desk, cup_or_food, ...};
- persons: PersonTracker output (pipeline/persons.py) or None: events, crowd_intervals, tracks;
- facts:   per-window transaction facts (pipeline/vlm_facts.py) or None;
- checks:  {"leave": [...], "entry": [...]} multi-frame event checks (pipeline/checks.py) or {}.

A candidate is {type_id, t, conf, observation, evidence_t: [t...], region}. `observation` is Russian text with what
is visible (no legal conclusions; analyze.py appends the norm). All thresholds live in PARAMS. They were set on the
42 open clips (team labels in analysis/gt_team.json); timecode conventions follow the organizer labels:
an action violation is stamped where the action starts, a "left behind on leaving" violation about 6 s before the
cashier exits (the moment she gets up and heads for the door).
"""
from __future__ import annotations

from dataclasses import dataclass, field

PARAMS = {
    "smooth": 3,  # majority window (frames at 1 fps) for boolean states
    "absent_min": 3,  # seconds of continuous absence that count as "the cashier left" (shorter at the clip end)
    "after_leave": 4,  # frames after the exit used to see what stayed on the desk / screen
    "leave_back_persons": 6.5,  # leave timecode = tracker exit - this
    "leave_back_states": 3.5,  # leave timecode = first VLM-absent frame - this (no tracker exit near it)
    "left_item_min_frac": 0.5,  # share of after-exit frames that must show the item
    "safe_open_tail_min": 3,  # safe open until the clip end for at least this many frames
    "safe_open_long": 10,  # or open at least this long while the cashier is away from it
    "personal_min": 6,  # seconds a personal item / cup must stay on the desk (appearing after the start)
    "personal_tail_min": 4,  # shorter run accepted when it lasts to the end of the clip
    "personal_offset": 2.0,  # T12 timecode = first frame with the item + offset (it is put down, then settles)
    "crowd_min": 1.0,  # seconds with more than 2 people in the room
    "crowd_offset": 4.0,  # T9 timecode = crowd start + offset (clipped to the interval)
    "session_gap": 3.0,  # VLM people-count gaps up to this long do not split a visitor session
    "t1_offset": 1.5,  # T1 timecode = first handover + offset (the cashier reaches for the money)
    "t1_tail": 7.0,  # the video must go on this long after the first handover to judge the ID handover
    "return_gap": 3.0,  # clustering gap for "notes back to the visitor" observations
    "machine_near": 2.0,  # ... which must coincide with counting-machine activity within this many seconds
    "uncounted_machine_max": 2,  # T13 bag: at most this many "into the machine" observations
    "typing_after_max": 8.0,  # T13 money: typing within this many seconds after the last handover
    "door_skip": 4.0,  # T8: seconds after an entry before the door must be closed
    "door_window": 10.0,  # T8: window after door_skip that is checked
    "door_open_frac": 0.6,  # T8: share of frames in the window with the door open
    "door_entry_offset": 3.5,  # T8 timecode = entry + offset (no stranger entering)
    "door_open_diff": 28.0,  # T8: settled door-crop difference to the shut reference that counts as an open leaf
    "door_min_after": 9.0,  # T8: seconds of video needed after an entry (closing takes up to 6.5 s + 1.5 s stable)
}


@dataclass
class Signals:
    duration: float
    states: list[dict]
    persons: dict | None = None
    checks: dict = field(default_factory=dict)
    facts: list[dict] | None = None


def mmss(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:04.1f}"


def _smooth(flags: list[bool], k: int) -> list[bool]:
    """Centered majority filter; k odd."""
    if k <= 1:
        return list(flags)
    h = k // 2
    out = []
    for i in range(len(flags)):
        w = flags[max(0, i - h): i + h + 1]
        out.append(sum(w) * 2 > len(w))
    return out


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """[(start, end_inclusive)] of True runs."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _clusters(ts: list[float], gap: float) -> list[list[float]]:
    out: list[list[float]] = []
    for t in sorted(ts):
        if out and t - out[-1][-1] <= gap:
            out[-1].append(t)
        else:
            out.append([t])
    return out


# ------------------------------------------------------------------ visitor sessions
@dataclass
class Session:
    t0: float
    t1: float


def visitor_sessions(sig: Signals, p: dict) -> list[Session]:
    """Intervals with a visitor in the room: VLM people count >= 2 (smoothed, short gaps bridged); the person
    tracker's visitor tracks when the VLM count gives nothing."""
    st = sig.states
    out: list[Session] = []
    if st:
        two = _smooth([(s.get("people_in_room") or 0) >= 2 for s in st], p["smooth"])
        for a, b in _runs(two):
            if out and st[a]["t"] - out[-1].t1 <= p["session_gap"]:
                out[-1].t1 = st[b]["t"]
            else:
                out.append(Session(st[a]["t"], st[b]["t"]))
    if not out and sig.persons:
        for tr in sig.persons.get("tracks") or []:
            if tr.get("role") == "visitor":
                out.append(Session(tr.get("t_room_start", tr["t_start"]), tr.get("t_room_end", tr["t_end"])))
    return sorted(out, key=lambda s: s.t0)


def _session_facts(sig: Signals, t0: float, t1: float) -> dict | None:
    if not sig.facts:
        return None
    from pipeline.vlm_facts import session_facts
    return session_facts(sig.facts, t0, t1)


def _confirmed_deposits(sig: Signals, into_safe: list[float], tol: float = 3.0) -> list[float]:
    """`into_safe` window observations backed by a per-frame VLM `safe=open` within tol seconds. Used only by the
    T2 / T11 rules: a stray into_safe fact there suppresses a true alarm (robustness report fix 3); applied
    globally it would break T13 on X-13, where the safe state itself is missed."""
    opened = [s["t"] for s in sig.states if s.get("safe") == "open"]
    return [t for t in into_safe if any(abs(t - u) <= tol for u in opened)]


# ------------------------------------------------------------------ leaving
@dataclass
class Leave:
    t: float  # timecode for "left behind" violations
    t_stand: float
    t_exit: float  # first VLM-absent frame
    i_exit: int  # index of that frame in states
    t_leave: float  # last moment the cashier is in the room (tracker event when available, else t_exit)


def find_leaves(sig: Signals, p: dict) -> list[Leave]:
    """The cashier leaves the room: a VLM-absent run after presence (>= absent_min s, or to the clip end).
    Times are refined by the person tracker's cashier_leave events when they agree."""
    st = sig.states
    if not st:
        return []
    present = _smooth([s.get("cashier") in ("seated", "standing") for s in st], p["smooth"])
    ev = (sig.persons or {}).get("events") or []
    p_leaves = [e for e in ev if e.get("type") == "cashier_leave"]
    p_stands = [e["t"] for e in ev if e.get("type") == "cashier_stand"]
    out = []
    for a, b in _runs([not x for x in present]):
        if a == 0:
            continue
        n = b - a + 1
        if n < p["absent_min"] and not (b == len(st) - 1 and n >= 2):
            continue
        t_exit = st[a]["t"]
        j = a - 1
        while j >= 0 and st[j].get("cashier") == "standing":
            j -= 1
        t_stand = st[min(j + 1, a - 1)]["t"]
        t = t_exit - p["leave_back_states"]
        t_leave = t_exit
        # the tracker keeps seeing the cashier when the room goes dark (light switched off, X-11): up to +7 s
        near = [e for e in p_leaves if -2.0 <= e["t"] - t_exit <= 7.0]
        if near:
            e = min(near, key=lambda e: abs(e["t"] - t_exit))
            t = e["t"] - p["leave_back_persons"]
            t_leave = e["t"]
            stands = [s for s in p_stands if e["t"] - 25.0 <= s <= e["t"]]
            if stands:
                t_stand = max(stands)
        t = max(t, min(t_stand, t_exit))
        out.append(Leave(t=round(t, 1), t_stand=t_stand, t_exit=t_exit, i_exit=a, t_leave=t_leave))
    return out


def _check_for(sig: Signals, kind: str, key: str, t: float, tol: float = 3.0) -> dict | None:
    """The multi-frame event check of `kind` ("leave" | "entry") whose `key` time is closest to t (within tol)."""
    cands = [c for c in (sig.checks or {}).get(kind) or [] if c.get(key) is not None and abs(c[key] - t) <= tol]
    return min(cands, key=lambda c: abs(c[key] - t)) if cands else None


def rule_leave_items(sig: Signals, p: dict) -> list[dict]:
    """T3 cash/bag, T7 notebook, T6 laptop left working when the cashier leaves.

    Two independent views, either one is enough (a miss is worse than a false alarm, FAQ 16): the per-frame scene
    states after the exit, and the leave check (pipeline/checks.py: one question per desk object on the empty room,
    the laptop screen before/after)."""
    out = []
    st = sig.states
    for lv in find_leaves(sig, p):
        after = [s for s in st[lv.i_exit: lv.i_exit + p["after_leave"]] if s.get("ok")]
        chk = _check_for(sig, "leave", "t_leave", lv.t_leave)
        if not after and chk is None:
            continue

        def frac(key, value=True):
            return sum(s.get(key) == value for s in after) / len(after) if after else 0.0

        left = (chk or {}).get("left_on_desk") or {}
        t_after = after[min(1, len(after) - 1)]["t"] if after else lv.t_leave + 2.0
        ev = [lv.t, lv.t_leave, t_after]
        base = f"Кассир встаёт и выходит из кабинета ({mmss(lv.t_leave)})."
        cash = frac("cash_on_desk") >= p["left_item_min_frac"] or bool(left.get("cash"))
        bag = frac("bag_on_desk") >= p["left_item_min_frac"] or bool(left.get("bag"))
        if cash or bag:
            what = "остаётся сумка (мешок) с деньгами" if bag and not cash else "остаются наличные деньги"
            out.append({"type_id": "T3", "t": lv.t, "conf": 0.75, "region": "desk", "evidence_t": ev,
                        "observation": f"{base} После ухода на столе {what}, в кабинете никого нет."})
        if frac("notebook_on_desk") >= p["left_item_min_frac"] or left.get("notebook_or_documents"):
            out.append({"type_id": "T7", "t": lv.t, "conf": 0.6, "region": "desk", "evidence_t": ev,
                        "observation": f"{base} После ухода на столе остаётся тетрадь (папка, документы)."})
        laptop_on = frac("laptop", "white_window") >= p["left_item_min_frac"]
        if chk is not None and chk.get("laptop_screen_after") not in (None, "not_visible"):
            laptop_on = laptop_on or chk["laptop_screen_after"] == "white_window"
        if laptop_on:
            out.append({"type_id": "T6", "t": lv.t, "conf": 0.7, "region": "laptop", "evidence_t": ev,
                        "observation": f"{base} После ухода на экране ноутбука остаётся открытое рабочее окно; "
                                       f"экрана блокировки не видно."})
    return out


def rule_key_left(sig: Signals, p: dict) -> list[dict]:
    """T5: the safe key stays in the lock when the cashier leaves (leave check key_left_in_safe: the key cord under the
    keyhole is seen while she works and still after she walked out). Timecode = the moment she is done with the lock
    (end of the last stretch with her hand over the keyhole), clipped to [t_stand, t_leave]."""
    for lv in find_leaves(sig, p):
        chk = _check_for(sig, "leave", "t_leave", lv.t_leave)
        if not chk or not chk.get("key_left_in_safe"):
            continue
        t_lock = chk.get("t_lock")
        t = round(min(max(t_lock, lv.t_stand), lv.t_leave), 1) if t_lock is not None else lv.t
        t_after = chk.get("evidence_t", lv.t_leave + 2.0)
        return [{"type_id": "T5", "t": t, "conf": 0.6, "region": "safe", "zoom": chk.get("lock_box"),
                 "evidence_t": [t, lv.t_leave, t_after],
                 "observation": f"Кассир закрывает сейф ({mmss(t)}) и выходит из кабинета ({mmss(lv.t_leave)}). "
                                f"Ключ остаётся в замке сейфа: на увеличенном фрагменте под замочной скважиной висит "
                                f"брелок (шнурок) ключа, как и до ухода."}]
    return []


# ------------------------------------------------------------------ safe, personal items, crowd
def rule_safe_open(sig: Signals, p: dict) -> list[dict]:
    """T4: the safe stays open while the cashier works at the desk or is away (to the clip end or for long).

    Besides clean runs, a flickering tail counts as "open to the end": from the first open frame the safe is open
    in >= 60 % of the remaining frames and in >= 3 of the last 5 (robustness report fix 4)."""
    st = sig.states
    raw_open = [s.get("safe") == "open" for s in st]
    open_ = _smooth(raw_open, p["smooth"])
    runs = list(_runs(open_))
    first = next((i for i, o in enumerate(raw_open) if o), None)
    n = len(st)
    if first is not None and n - first >= 5:
        rest = raw_open[first:]
        if sum(rest) >= 0.6 * len(rest) and sum(raw_open[-5:]) >= 3 and not any(b == n - 1 for _, b in runs):
            runs.append((first, n - 1))
    out = []
    for a, b in runs:
        away = [i for i in range(a, b + 1) if st[i].get("cashier") in ("seated", "absent")]
        long_enough = len(away) >= p["safe_open_long"]
        to_end = b == n - 1 and len(away) >= p["safe_open_tail_min"]
        if not (long_enough or to_end):
            continue
        t = st[away[0]]["t"]
        out.append({"type_id": "T4", "t": t, "conf": 0.65, "region": "safe",
                    "evidence_t": [st[a]["t"], t, st[b]["t"]],
                    "observation": f"Дверца сейфа открыта с {mmss(st[a]['t'])}; кассир отходит от сейфа и продолжает "
                                   f"работу, сейф остаётся открытым"
                                   + (" до конца записи." if to_end else f" до {mmss(st[b]['t'])}.")})
    return sorted(out, key=lambda c: c["t"])[:1]


def rule_personal_items(sig: Signals, p: dict) -> list[dict]:
    """T12: clothing or a cup / food appears on the desk and stays there."""
    st = sig.states
    out = []
    for key, what in (("clothing_on_desk", "верхняя одежда (куртка) или личная сумка"),
                      ("cup_or_food", "чашка (напиток) или еда")):
        flags = _smooth([bool(s.get(key)) for s in st], p["smooth"])
        for a, b in _runs(flags):
            if a <= 1:  # present from the start: part of the scene, not brought in
                continue
            n = b - a + 1
            if n >= p["personal_min"] or (b == len(st) - 1 and n >= p["personal_tail_min"]):
                t0 = st[a]["t"]
                t = round(min(t0 + p["personal_offset"], st[b]["t"]), 1)
                out.append({"type_id": "T12", "t": t, "conf": 0.55, "region": "desk",
                            "evidence_t": [t0, t, st[min(b, a + 4)]["t"]],
                            "observation": f"С {mmss(t0)} на рабочем столе кассира находится {what}."})
                break
    return sorted(out, key=lambda c: c["t"])[:1]


def rule_crowd(sig: Signals, p: dict) -> list[dict]:
    """T9: more than two people in the room at once: tracker intervals (VLM-verified) or the VLM count itself."""
    intervals = [list(x) + [0.75] for x in (sig.persons or {}).get("crowd_intervals") or []]
    st = sig.states
    if st:
        flags = _smooth([(s.get("people_in_room") or 0) >= 3 for s in st], p["smooth"])
        for a, b in _runs(flags):
            if b - a + 1 >= 2 and not any(x[0] - 2 <= st[a]["t"] <= x[1] + 2 for x in intervals):
                n = max(s.get("people_in_room") or 0 for s in st[a:b + 1])
                intervals.append([st[a]["t"], st[b]["t"], n, 0.6])
    out = []
    for t0, t1, n, conf in intervals:
        if t1 - t0 < p["crowd_min"]:
            continue
        t = round(min(t0 + p["crowd_offset"], t1), 1)
        out.append({"type_id": "T9", "t": t, "conf": conf, "region": "persons", "evidence_t": [t0, t, t1],
                    "observation": f"С {mmss(t0)} по {mmss(t1)} в кабинете одновременно находятся {int(n)} человека "
                                   f"(люди за стеклом и отражения не учитываются)."})
    return out[:1]


# ------------------------------------------------------------------ transaction rules (facts)
def rule_no_id(sig: Signals, p: dict) -> list[dict]:
    """T1: the visitor hands over money once, with no separate ID handover before it; T10 add-on when the completed
    reception is not recorded."""
    out = []
    for s in visitor_sessions(sig, p):
        sf = _session_facts(sig, s.t0 - 1.0, s.t1 + 1.0)
        if not sf or not sf["handovers"]:
            continue
        th = sf["first_handover"]
        # judge only if the video goes on long enough after the handover (an ID could still follow);
        # a visitor who leaves right after handing over (X-17) is judged anyway
        too_short = th + p["t1_tail"] > sig.duration + 1.0 and s.t1 >= sig.duration - 2.0
        if too_short or sf["handovers_first10"] != 1:
            continue
        t = round(th + p["t1_offset"], 1)
        bag = any(x.get("bag_on_desk") for x in sig.states if th - 1.0 <= x["t"] <= th + 4.0)
        what = "сумку (мешок) с деньгами" if bag else "деньги"
        out.append({"type_id": "T1", "t": t, "conf": 0.6, "region": "desk", "evidence_t": [th, t, t + 2.0],
                    "observation": f"Посетитель передаёт кассиру {what}, кассир их принимает ({mmss(t)}). До этого не "
                                   f"видно, чтобы посетитель передавал или предъявлял удостоверение личности."})
        safe = sf["money_to_safe"]["t"] if sf["money_to_safe"]["value"] else None
        left = s.t1 < sig.duration - 2.0
        if not sf["recorded"] and (safe is not None or left):
            t10 = round(safe if safe is not None else s.t1, 1)
            what = "деньги убраны в сейф" if safe is not None else "посетитель уходит"
            out.append({"type_id": "T10", "t": t10, "conf": 0.4, "region": "desk", "evidence_t": [t10],
                        "observation": f"Приём денег завершён ({what}, {mmss(t10)}), но не видно, чтобы кассир "
                                       f"вносил операцию в ноутбук или журнал."})
        break
    return out


def rule_next_visitor_cash(sig: Signals, p: dict) -> list[dict]:
    """T2: the next visitor comes in while the previous visitor's money has not gone into the safe."""
    sess = visitor_sessions(sig, p)
    st = sig.states
    for prev, cur in zip(sess, sess[1:]):
        sf = _session_facts(sig, prev.t0 - 1.0, cur.t0)
        safe_seen = any(s.get("safe") == "open" for s in st if prev.t0 <= s["t"] <= cur.t0)
        if sf is not None:
            taken = bool(sf["handovers"]) or sf["cash_received"]["votes"] >= 1
            deposits = _confirmed_deposits(sig, sf["timeline"]["into_safe"])
            if not taken or deposits or safe_seen:
                continue
            th = sf["first_handover"] if sf["first_handover"] is not None else prev.t0
        else:
            win = [s for s in st if cur.t0 - 2.0 <= s["t"] <= cur.t0 + 3.0 and s.get("ok")]
            if not win or sum(bool(s.get("cash_on_desk")) for s in win) * 2 <= len(win) or safe_seen:
                continue
            th = prev.t0
        t = round(cur.t0, 1)
        return [{"type_id": "T2", "t": t, "conf": 0.65, "region": "desk", "evidence_t": [t - 2.0, t, t + 2.0],
                 "observation": f"В {mmss(t)} в кабинет входит следующий посетитель. Деньги, полученные от "
                                f"предыдущего посетителя ({mmss(th)}), к этому моменту в сейф не убраны: не видно, "
                                f"чтобы сейф открывался."}]
    return []


def rule_uncounted_bag(sig: Signals, p: dict) -> list[dict]:
    """T13: a cash bag is opened and the money goes into the safe with (almost) no machine counting; T10 add-on."""
    sf = _session_facts(sig, float("-inf"), float("inf"))
    if not sf or not sf["bag_opened"]["value"] or not sf["money_to_safe"]["value"]:
        return []
    if sf["machine_votes"] > p["uncounted_machine_max"]:
        return []
    safe_ts = sf["timeline"]["into_safe"]
    best = max(_clusters(safe_ts, 4.0), key=len) if safe_ts else []
    if not best:
        return []
    t = round(best[0], 1)
    tb = sf["bag_opened"]["t"]
    out = [{"type_id": "T13", "t": t, "conf": 0.55, "region": "safe", "evidence_t": [tb, t, t + 2.0],
            "observation": f"Кассир вскрывает сумку с деньгами ({mmss(tb)}) и убирает деньги в сейф ({mmss(t)}); не "
                           f"видно, чтобы деньги пересчитывались на счётчике банкнот."}]
    if not sf["recorded"]:
        out.append({"type_id": "T10", "t": t, "conf": 0.4, "region": "desk", "evidence_t": [t],
                    "observation": f"Деньги из сумки убраны в сейф ({mmss(t)}), но не видно, чтобы кассир вносил "
                                   f"операцию в ноутбук или журнал."})
    return out


def rule_typed_without_count(sig: Signals, p: dict) -> list[dict]:
    """T13: within a visit the cashier takes the money and goes straight to typing it in, with no counting-machine
    activity between the last handover and the typing (X-04/X-47, second visitor)."""
    if not sig.facts:
        return []
    for s in visitor_sessions(sig, p):
        sf = _session_facts(sig, s.t0 - 1.0, s.t1 + 1.0)
        if not sf or not sf["handovers"]:
            continue
        tl = sf["timeline"]
        for t_typ in tl["typing"]:
            before = [h for h in sf["handovers"] if h + 2.0 < t_typ]
            if not before:
                continue
            th = before[-1]
            if t_typ - th > p["typing_after_max"]:
                continue
            if any(th - 1.0 <= m <= t_typ for m in tl["into_machine"] + tl["band_removed"]):
                continue
            t = round(max(th, t_typ - 2.0), 1)
            return [{"type_id": "T13", "t": t, "conf": 0.5, "region": "desk", "evidence_t": [th, t, t_typ],
                     "observation": f"Кассир получает деньги от посетителя ({mmss(th)}) и сразу вносит операцию в "
                                    f"ноутбук ({mmss(t_typ)}); не видно, чтобы деньги пересчитывались на счётчике "
                                    f"банкнот."}]
    return []


def rule_return_notes(sig: Signals, p: dict) -> list[dict]:
    """T11: after machine counting the cashier hands notes back to the visitor (not the end-of-visit ID return)."""
    if not sig.facts:
        return []
    for s in visitor_sessions(sig, p):
        sf = _session_facts(sig, s.t0 - 1.0, s.t1 + 1.0)
        if not sf:
            continue
        tl = sf["timeline"]
        mach, safe = tl["into_machine"], _confirmed_deposits(sig, tl["into_safe"])
        for c in _clusters(tl["money_back_to_visitor"], p["return_gap"]):
            if len(c) < 2:
                continue
            t = c[0]
            if not any(m < t for m in mach) or not any(abs(m - u) <= p["machine_near"] for m in mach for u in c):
                continue
            if any(x < t for x in safe):
                continue
            t = round(t, 1)
            return [{"type_id": "T11", "t": t, "conf": 0.55, "region": "desk", "evidence_t": [t - 1.0, t, t + 2.0],
                     "observation": f"После пересчёта на счётчике банкнот кассир возвращает посетителю часть купюр "
                                    f"({mmss(t)}). Не видно, чтобы на возвращаемые купюры оформлялся акт."}]
    return []


def entries(sig: Signals) -> list[tuple[float, str]]:
    """[(t, who)] entries through the door: the cashier coming in (tracker cashier_return, or the VLM cashier
    absent -> present at the start) and visitors coming in (tracker visitor_enter)."""
    ev = (sig.persons or {}).get("events") or []
    out = [(e["t"], "cashier") for e in ev if e.get("type") == "cashier_return"]
    st = sig.states
    if not out and st and st[0].get("cashier") == "absent":
        for s in st:
            if s.get("cashier") in ("seated", "standing"):
                out.append((s["t"], "cashier"))
                break
    out += [(e["t"], "visitor") for e in ev if e.get("type") == "visitor_enter"]
    return sorted(out)


def door_ref_time(sig: Signals, t_enter: float, pre: float = 3.0) -> float | None:
    """Reference moment for the entry door check: None (= t_enter - pre) unless another visitor left shortly
    before this entry; then the door may be in use at t_enter - pre, so take the middle of that previous visit,
    when the door was shut with the visitor inside."""
    ev = (sig.persons or {}).get("events") or []
    exits = [e["t"] for e in ev if e.get("type") == "visitor_exit" and t_enter - 8.0 <= e["t"] <= t_enter]
    if not exits:
        return None
    t_exit = max(exits)
    starts = [e["t"] for e in ev if e.get("type") == "visitor_enter" and e["t"] < t_exit - 6.0]
    if not starts:
        return None
    t0 = max(starts)
    return round(t0 + min(10.0, (t_exit - t0) / 2), 1)


def rule_door_open(sig: Signals, p: dict) -> list[dict]:
    """T8: after someone comes in, the door is not shut again while the cashier works.

    Primary signal: the entry check (pipeline/checks.py: door-crop pixel difference to a frame before the entry;
    12/12 on the open clips). Fallback without a check: VLM door states (unreliable on some camera positions)."""
    st = sig.states
    visitors = [e["t"] for e in ((sig.persons or {}).get("events") or []) if e.get("type") == "visitor_enter"]
    for te, who in entries(sig):
        if sig.duration - te < p["door_min_after"]:
            continue
        chk = _check_for(sig, "entry", "t_enter", te, tol=1.0)
        watched = None
        if chk is not None:
            if chk.get("door_closed_after_entry"):
                continue
            raw = chk.get("raw") or {}
            # the door crop never changed although someone came in: the crop missed the door (rotated / dark
            # camera, robustness report fix 2) or the door stood open; inconclusive, no alarm
            if raw.get("t_opened") is None:
                continue
            # how far the door crop stays from the shut reference once the entry is over: an open leaf keeps it at
            # 31-41 (X-30, X-32, X-33); smaller values are a shut door with a seated visitor in the crop (X-39 24.5,
            # X-17 23.3), not an open door
            tail = [d for t, d in (raw.get("door_diff") or []) if t >= te + 6.0][-4:]
            if tail and sum(tail) / len(tail) < p["door_open_diff"]:
                continue
            if chk.get("someone_else_entered_or_watched") and chk.get("t_other"):
                watched = chk["t_other"]
        else:
            after = [s for s in st if s["t"] > te + 1.0]
            door = [s.get("door") for s in after]
            closed_at = next((after[k]["t"] for k in range(len(door) - 2)
                              if door[k] == door[k + 1] == door[k + 2] == "closed"), None)
            if closed_at is not None and closed_at <= te + p["door_skip"] + p["door_window"]:
                continue
            n_open = sum(d == "open" for s, d in zip(after, door) if s["t"] >= te + p["door_skip"])
            if n_open < (2 if who == "cashier" else 4):
                continue
        # timecode: a stranger walking in (tracker visitor_enter) marks the moment; someone only watching through
        # the doorway is mentioned but keeps the entry-based timecode (organizer labels: X-30 13 s, X-33 6 s)
        walk_in = [v for v in visitors if te + 2.0 < v <= te + 25.0]
        if walk_in:
            t = round(walk_in[0] + 0.5, 1)
            extra = f" В {mmss(walk_in[0])} в открытую дверь входит посторонний человек."
        else:
            t = round(te + p["door_entry_offset"], 1)
            extra = (f" В {mmss(watched)} за открытой дверью виден посторонний человек, наблюдающий за работой "
                     f"кассира." if watched is not None else "")
        actor = "Кассир входит" if who == "cashier" else "Посетитель входит"
        return [{"type_id": "T8", "t": t, "conf": 0.65 if chk is not None else 0.5, "region": "door",
                 "evidence_t": [te + 1.0, t, watched if watched is not None else t + 3.0],
                 "observation": f"{actor} в кабинет ({mmss(te)}), дверь кассы после этого не закрывается и остаётся "
                                f"открытой, пока кассир работает.{extra}"}]
    return []


RULES = [rule_crowd, rule_leave_items, rule_key_left, rule_safe_open, rule_personal_items, rule_no_id,
         rule_next_visitor_cash, rule_uncounted_bag, rule_typed_without_count, rule_return_notes, rule_door_open]


def evaluate(sig: Signals, params: dict | None = None) -> list[dict]:
    """Candidates sorted by confidence (most reliable first), one per type."""
    p = {**PARAMS, **(params or {})}
    cands = []
    for r in RULES:
        cands.extend(r(sig, p))
    best: dict[str, dict] = {}
    for c in cands:
        k = c["type_id"]
        if k not in best or (c["conf"], -c["t"]) > (best[k]["conf"], -best[k]["t"]):
            best[k] = c
    return sorted(best.values(), key=lambda c: (-c["conf"], c["t"]))
