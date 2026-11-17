"""Step 5: who owns the ball -- touches -> close / deflect / own events -> a raw H/A label per frame.

Inputs: out/clean_table.csv, out/touches.csv, out/ball_trips.csv, out/keepers.csv, and player
positions from the raw tracking (clean_table.read_tracking).

- Team of a slot on a frame: the keeper's team if keepers.csv names the slot as a keeper on that
  frame (team_source = keeper), otherwise its nominal team (team_source = nominal). Frozen and absent
  slots are never candidates. The ball position is the step-1 ball with glitches (speed >
  SANE_BALL_SPEED) masked, as in step 4.
- At each touch, at onset_frame: every usable player within CONTROL_RADIUS gets a `close` event; the
  nearest of them is the toucher (none within the radius -> no toucher -> hold).
  * speed_up (a kick/pass/shot): `own` for the kicker's team at onset (reason own_kick).
  * otherwise `own` if the toucher stays the nearest player, or within CONTROL_RADIUS, for at least
    MIN_OWN_FRAMES consecutive ball-seen frames before the next touch's onset, and before the ball
    leaves him at speed (>= FAST_BALL after having been slower); missing-ball frames and frames
    where the ball is still arriving at >= FAST_BALL neither count nor break the run. Else `deflect`.
  * The toucher is searched over onset +-TOUCH_ATTRIB_FRAMES: the frame where the nearest usable
    player is closest to the ball (within CONTROL_RADIUS); `close`, the toucher and his team are
    taken on that frame, and the own check starts there. The label still changes at the onset.
  * Touches sharing an onset (a keeper's throw is a stop_go and a speed_up at the same minimum) are
    one event: speed_up, then stop_go, then speed_down, then turn.
  * A keeper's own whose keeper team differs from the slot's nominal team has reason keeper_fix.
- Long gap: at every trip with starts_with = gap, the first player who is the nearest within
  CONTROL_RADIUS for MIN_OWN_FRAMES consecutive frames (before the trip's first touch onset) is an
  `own` at the first of those frames (reason gap_reappear). Nobody -> hold.
- The label changes only on `own`, at its onset frame. Between owns it holds; each held frame gets
  the reason that applies, in this order: hold_missing (no ball), hold_fast_ball (>= FAST_BALL),
  hold_close (players of both teams within CONTROL_RADIUS), hold_deflect (the last event since the
  own was a deflection), hold_no_candidate (nobody within CONTROL_RADIUS), else the own's reason.
- Start of a half: until the first own, the team of the first player within CONTROL_RADIUS of the
  ball (reason start_of_half); if there is none, the label stays empty for steps 6/7.

Outputs: out/events.csv, out/labels_raw.csv. Stdout: aggregates only.

    python src/possession.py
    python src/possession.py --show <vf> ... --around 25     # per-frame rows for local debugging
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = next((d for d in (HERE, os.path.dirname(HERE), os.getcwd())
             if os.path.isfile(os.path.join(d, "games.py"))), None)
if REPO is None:
    sys.exit("Can't find games.py. Run this from the repo root: python src/possession.py")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import games                                            # noqa: E402
from starter.baseline import team_of_each_track, HOME   # noqa: E402
import timeutil as T                                    # noqa: E402
import config as C                                      # noqa: E402
from clean_table import read_tracking, frozen_mask      # noqa: E402
from team_check import home_defends_first_half, LEFT    # noqa: E402

EVENT_COLUMNS = ["video_frame", "onset_frame", "slot", "team", "event", "dist_m", "team_source",
                 "reason"]
TEAMS = np.array(["H", "A"])


def fail_missing(name, path, hint=""):
    sys.exit("Can't find the %s file (%s). %s" % (name, path, hint or
             "Put this script next to games.py, or pass --%s /path/to/file." % name))


def load(p):
    """Everything per graded frame, as arrays indexed by row (row = position in clean_table)."""
    ct = pd.read_csv(p.table, usecols=["video_frame", "half", "ball_x_m", "ball_y_m", "ball_speed"])
    vf = ct["video_frame"].to_numpy()
    row_of = {v: i for i, v in enumerate(vf.tolist())}
    speed = ct["ball_speed"].to_numpy()
    glitch = speed > C.SANE_BALL_SPEED
    bx = np.where(glitch, np.nan, ct["ball_x_m"].to_numpy())
    by = np.where(glitch, np.nan, ct["ball_y_m"].to_numpy())
    speed = np.where(glitch, np.nan, speed)

    paths = games.asset_paths(source="tactical")
    teams_nominal = team_of_each_track(paths["atd_meta"])
    n_slots = len(teams_nominal)
    n_video = games.source_frames("tactical")
    margin = C.FROZEN_FRAMES - 1
    wanted = np.zeros(n_video, dtype=bool)
    for _, first, last in T.graded_blocks():
        wanted[max(0, first - margin):min(n_video, last + margin + 1)] = True
    tvf, players, _, _ = read_tracking(paths["atd_raw"], wanted, n_slots)
    usable_all = ~np.isnan(players[:, :, 0]) & ~frozen_mask(tvf, players, C.FROZEN_FRAMES)
    pos = np.searchsorted(tvf, vf)
    if not np.array_equal(tvf[pos], vf):
        sys.exit("Tracking does not cover every frame of the clean table.")
    players, usable = players[pos], usable_all[pos]

    # team per (frame, slot): nominal, overridden by keepers.csv
    nominal = np.where(teams_nominal == HOME, 0, 1).astype(np.int8)
    team = np.tile(nominal, (len(vf), 1))
    from_keeper = np.zeros_like(usable)
    kp = pd.read_csv(p.keepers).set_index("video_frame").reindex(vf)
    for code, col in ((0, "h_keeper_slot"), (1, "a_keeper_slot")):
        k = kp[col].to_numpy()
        rows = np.flatnonzero(~np.isnan(k))
        slots = k[rows].astype(int) - 1
        team[rows, slots] = code
        from_keeper[rows, slots] = True

    d = np.hypot(players[:, :, 0] - bx[:, None], players[:, :, 1] - by[:, None])
    d = np.where(usable & ~np.isnan(d), d, np.inf)
    nearest = np.argmin(d, axis=1)
    nd = d[np.arange(len(vf)), nearest]

    touches = pd.read_csv(p.touches)
    trips = pd.read_csv(p.trips)
    return {"vf": vf, "row_of": row_of, "half": ct["half"].to_numpy(), "bx": bx, "by": by,
            "speed": speed, "d": d, "nearest": nearest, "nd": nd, "team": team,
            "from_keeper": from_keeper, "nominal": nominal, "touches": touches, "trips": trips,
            "kp": kp}


def own_run(D, toucher, start, stop, radius, min_own):
    """(owned, max ball speed seen in the run) -- owned if the toucher stays nearest (or within
    radius) for min_own ball-seen frames in [start, stop) before the ball leaves him at speed."""
    streak, slowed, top = 0, False, 0.0
    for j in range(start, stop):
        if np.isnan(D["bx"][j]):
            continue                                          # missing ball: neither counts nor breaks
        sp = D["speed"][j]
        if not np.isnan(sp):
            if sp < C.FAST_BALL:
                slowed = True
            elif slowed:
                return False, top                             # the ball left him at speed
            else:
                continue                                      # still arriving at speed: not control yet
        if (D["nearest"][j] == toucher and np.isfinite(D["nd"][j])) or D["d"][j, toucher] <= radius:
            streak += 1
            if not np.isnan(sp):
                top = max(top, sp)
            if streak >= min_own:
                return True, top
        else:
            return False, top
    return False, top


# One touch per (half, onset): a keeper's throw is a stop_go and a speed_up at the same minimum.
ONSET_PRIORITY = {"speed_up": 0, "stop_go": 1, "speed_down": 2, "turn": 3}


def attribution_row(D, o, half, radius):
    """Frame on which the toucher is identified, or None.

    The toucher is the player at the ball when it is touched, not a challenger who arrives after:
    the onset frame if any usable player is within `radius` there; otherwise search outward
    (onset +-1, then +-2, ... up to TOUCH_ATTRIB_FRAMES) and take the first distance at which someone
    qualifies (if both sides do, the side whose nearest player is closer; then the earlier frame).
    """
    for k in range(C.TOUCH_ATTRIB_FRAMES + 1):
        rows = [r for r in ((o,) if k == 0 else (o - k, o + k))
                if 0 <= r < len(D["vf"]) and D["half"][r] == half and D["nd"][r] <= radius]
        if rows:
            return min(rows, key=lambda r: (D["nd"][r], r))
    return None


def touch_events(D, p):
    """events, owns [(onset_row, team, reason, video_frame)] and counters from the touches."""
    events, owns = [], []
    stats = {"no_toucher": 0, "same_onset_dropped": 0, "attributed": {}, "touches": {},
             "fast_own_touch": 0}
    tt = D["touches"].copy()
    tt["prio"] = tt["kind"].map(ONSET_PRIORITY)
    tt = tt.sort_values(["half", "onset_frame", "prio", "video_frame"])
    stats["same_onset_dropped"] = int(tt.duplicated(["half", "onset_frame"]).sum())
    tt = tt.drop_duplicates(["half", "onset_frame"]).sort_values("video_frame").reset_index(drop=True)
    onset_rows = [D["row_of"].get(v) for v in tt["onset_frame"].tolist()]
    for k, t in tt.iterrows():
        kind = t["kind"]
        stats["touches"][kind] = stats["touches"].get(kind, 0) + 1
        o = onset_rows[k]
        if o is None:
            stats["no_toucher"] += 1
            continue
        a = attribution_row(D, o, t["half"], p.attrib_radius)
        if a is None:
            stats["no_toucher"] += 1
            continue
        stats["attributed"][kind] = stats["attributed"].get(kind, 0) + 1
        # the next touch's onset in the same half bounds the own check
        nxt = next((onset_rows[m] for m in range(k + 1, len(tt))
                    if tt["half"].iat[m] == t["half"] and onset_rows[m] is not None
                    and onset_rows[m] > o), None)
        half_end = a + 1 + int(np.searchsorted(D["half"][a + 1:], t["half"], side="right"))
        stop = min(nxt, half_end) if nxt is not None else half_end
        vf = int(t["video_frame"])
        for s in np.flatnonzero(D["d"][a] <= p.radius):
            events.append(event(D, a, vf, s, "close", "within_radius", o))
        s = int(D["nearest"][a])
        if kind == "speed_up":
            reason = "own_kick"
        else:
            owned, top = own_run(D, s, a, max(stop, a + 1), p.attrib_radius, p.min_own)
            if not owned:
                events.append(event(D, a, vf, s, "deflect", "hold_deflect", o))
                continue
            reason = "own_touch"
            if top >= C.FAST_BALL:
                stats["fast_own_touch"] += 1
        if D["from_keeper"][a, s] and D["team"][a, s] != D["nominal"][s]:
            reason = "keeper_fix"
        events.append(event(D, a, vf, s, "own", reason, o))
        owns.append((o, int(D["team"][a, s]), reason, vf))
    return events, owns, stats


def gap_events(D, p, touch_onsets):
    """own events where the ball reappears after a long gap (trips with starts_with = gap)."""
    events, owns = [], []
    onsets = np.array(sorted(touch_onsets), dtype=np.int64)
    for _, t in D["trips"][D["trips"]["starts_with"] == "gap"].iterrows():
        a, b = D["row_of"].get(int(t["start_frame"])), D["row_of"].get(int(t["end_frame"]))
        if a is None or b is None:
            continue
        nxt = onsets[np.searchsorted(onsets, a):]
        stop = min(b + 1, int(nxt[0])) if len(nxt) else b + 1
        cur, start, streak = -1, None, 0
        for j in range(a, stop):
            s = int(D["nearest"][j]) if D["nd"][j] <= p.radius else -1
            if s >= 0 and s == cur:
                streak += 1
            elif s >= 0:
                cur, start, streak = s, j, 1
            else:
                cur, start, streak = -1, None, 0
            if streak >= p.min_own:
                events.append(event(D, start, int(D["vf"][start]), cur, "own", "gap_reappear"))
                owns.append((start, int(D["team"][start, cur]), "gap_reappear",
                             int(D["vf"][start])))
                break
    return events, owns


def event(D, row, frame, slot, kind, reason, onset_row=None):
    """Event row; team and distance measured on `row` (the attribution frame)."""
    o = row if onset_row is None else onset_row
    return {"video_frame": frame, "onset_frame": int(D["vf"][o]), "slot": int(slot) + 1,
            "team": TEAMS[D["team"][row, slot]], "event": kind,
            "dist_m": float(D["d"][row, slot]),
            "team_source": "keeper" if D["from_keeper"][row, slot] else "nominal", "reason": reason}


def ball_out_rows(D):
    """[n] True from a ball-out sighting until the first touch onset after the ball is seen back
    inside -- step 6's ball-out D stretches, rebuilt here because step 6 runs after this step."""
    from dead_ball import out_reasons
    reasons = out_reasons(D)
    n = len(D["vf"])
    seen = ~np.isnan(D["bx"])
    inside = seen & (reasons == "")
    onsets = np.array(sorted({D["row_of"][v] for v in D["touches"]["onset_frame"] if v in D["row_of"]}),
                      dtype=np.int64)
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if not reasons[i]:
            i += 1
            continue
        h = D["half"][i]
        b = i + int(np.searchsorted(D["half"][i:], h, side="right"))
        back = i + 1 + int(np.argmax(inside[i + 1:b])) if inside[i + 1:b].any() else b
        k = np.searchsorted(onsets, back)
        end = int(onsets[k]) if k < len(onsets) and onsets[k] < b else b
        out[i:max(end, i + 1)] = True
        i = max(end, back, i + 1)
    return out


def box_distribution(D, ev, p, home_goal_h1):
    """Team from behaviour: holds in a penalty area that end in a long distribution away from goal.

    A hold = the ball seen inside one penalty area at < DIST_HOLD_SPEED for >= DIST_HOLD_FRAMES
    (missing-ball gaps up to DIST_HOLD_MAX_GAP_FRAMES don't end it), ending with a speed_up or stop_go
    touch inside that penalty area (onset up to TOUCH_WINDOW_FRAMES after the hold), whose trip travels
    >= DIST_MIN_TRIP_M and ends >= DIST_GAIN_M further from that goal's centre than it started. Then
    every own from the hold start to that touch's onset gets the team defending that goal (reason
    box_distribution, team_source behaviour), and an own is added at the hold start (and at the touch
    if it had none). Holds that start inside a ball-out stretch are skipped (never inside D).
    Returns (events, holds for --show, stats).
    """
    n = len(D["vf"])
    with np.errstate(invalid="ignore"):
        in_y = np.abs(D["by"] - C.PITCH_W / 2) <= C.PENALTY_AREA_WIDTH_M / 2
        side = np.where(in_y & (D["bx"] <= C.PENALTY_AREA_DEPTH_M), 0,
                        np.where(in_y & (D["bx"] >= C.PITCH_L - C.PENALTY_AREA_DEPTH_M), 1, -1))
        slow = ~(D["speed"] >= C.DIST_HOLD_SPEED)                       # NaN speed counts as slow
    seen = ~np.isnan(D["bx"])
    ok = seen & (side >= 0) & slow
    dead = ball_out_rows(D)

    tt = D["touches"].copy()
    tt["prio"] = tt["kind"].map(ONSET_PRIORITY)
    tt = tt.sort_values(["half", "onset_frame", "prio", "video_frame"]).drop_duplicates(
        ["half", "onset_frame"])
    tt = tt[tt["kind"].isin(["speed_up", "stop_go"])]
    t_rows = [(D["row_of"][o], int(v), k) for o, v, k in
              zip(tt["onset_frame"], tt["video_frame"], tt["kind"]) if o in D["row_of"]]
    t_rows.sort()
    # football touches that end a played ball: any touch WITH a toucher, or any own. A touch with no
    # toucher (a thrown or kicked ball slowing in flight, a tracking artifact) does not end it.
    all_tt = D["touches"].copy()
    all_tt["prio"] = all_tt["kind"].map(ONSET_PRIORITY)
    all_tt = all_tt.sort_values(["half", "onset_frame", "prio"]).drop_duplicates(["half", "onset_frame"])
    real = sorted(D["row_of"][o] for o, h in zip(all_tt["onset_frame"], all_tt["half"])
                  if o in D["row_of"] and attribution_row(D, D["row_of"][o], h, p.attrib_radius) is not None)
    real = sorted(set(real) | {D["row_of"][e_["onset_frame"]] for e_ in ev
                               if e_["event"] == "own" and e_["onset_frame"] in D["row_of"]})
    real = np.array(real, dtype=np.int64)

    def flight_end(o):
        """(end row, how) of the ball played at onset row o: the next real touch or own, a missing-
        ball gap longer than MAX_GAP_FRAMES, a ball-out stretch, the half end, or DIST_MAX_FLIGHT_FRAMES."""
        stop = min(o + C.DIST_MAX_FLIGHT_FRAMES, n - 1)
        k = int(np.searchsorted(real, o, side="right"))
        if k < len(real) and real[k] <= stop:
            stop, how = int(real[k]), "next touch"
        else:
            how = "4 s"
        miss = 0
        for r in range(o + 1, stop + 1):
            if D["half"][r] != D["half"][o]:
                return r - 1, "half end"
            if dead[r]:
                return r, "ball out"
            miss = miss + 1 if not seen[r] else 0
            if miss > C.MAX_GAP_FRAMES:
                return r - miss, "long gap"
        return stop, how

    holds, i = [], 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        s, last, sd = i, i, side[i]
        j = i + 1
        while j < n and D["half"][j] == D["half"][s]:
            if ok[j] and side[j] == sd:
                last = j
            elif seen[j] or j - last > C.DIST_HOLD_MAX_GAP_FRAMES:
                break
            j += 1
        if last - s + 1 >= C.DIST_HOLD_FRAMES:
            holds.append((s, last, int(sd)))
        i = last + 1

    stats = {"fired": {}, "changed": 0, "unchanged": 0, "no_toucher": 0, "keeper_agree": 0,
             "keeper_disagree": 0, "skipped_dead": 0}
    shown = []
    for s, last, sd in holds:
        h = int(D["half"][s])
        home_left = (home_goal_h1 == LEFT) == (h == 1)
        defend = 0 if (sd == 0) == home_left else 1
        gx = 0.0 if sd == 0 else C.PITCH_L
        info = {"start": s, "end": last, "side": sd, "team": defend, "fired": False, "why": ""}
        shown.append(info)
        if dead[s]:
            info["why"] = "inside a ball-out stretch"
            stats["skipped_dead"] += 1
            continue
        cands = [t for t in t_rows if s <= t[0] <= last + C.TOUCH_WINDOW_FRAMES and side[t[0]] == sd]
        if not cands:
            info["why"] = "no speed_up/stop_go touch in the box at the end"
            continue
        o, vf, kind = cands[-1]
        e, how = flight_end(o)
        seen_rows = [r for r in range(o, e + 1) if seen[r]]
        p0, p1 = (D["bx"][o], D["by"][o]), (D["bx"][seen_rows[-1]], D["by"][seen_rows[-1]])
        travel = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        gain = float(np.hypot(p1[0] - gx, p1[1] - C.PITCH_W / 2) - np.hypot(p0[0] - gx, p0[1] - C.PITCH_W / 2))
        info.update({"onset": o, "travel": travel, "gain": gain, "flight_end": seen_rows[-1],
                     "flight_how": how})
        if travel < C.DIST_MIN_TRIP_M or gain < C.DIST_GAIN_M:
            info["why"] = "trip %.0f m, gain %.0f m: too short or not away from goal" % (travel, gain)
            continue
        info["fired"] = True
        key = (h, TEAMS[defend])
        stats["fired"][key] = stats["fired"].get(key, 0) + 1
        # agreement / change checks on the distributing touch's toucher
        a = attribution_row(D, o, h, p.attrib_radius)
        if a is None:
            stats["no_toucher"] += 1
        else:
            slot = int(D["nearest"][a])
            if D["team"][a, slot] != defend:
                stats["changed"] += 1
            else:
                stats["unchanged"] += 1
            col = "h_keeper_slot" if defend == 0 else "a_keeper_slot"
            k = D["kp"][col].iloc[a]
            if not pd.isna(k) and int(k) - 1 == slot:
                stats["keeper_agree"] += 1
            else:
                stats["keeper_disagree"] += 1
        # override owns from the hold start to the distributing touch's onset
        lo, hi = int(D["vf"][s]), int(D["vf"][o])
        for e_ in ev:
            if e_["event"] == "own" and lo <= e_["onset_frame"] <= hi:
                stats["owns_changed" if e_["team"] != TEAMS[defend] else "owns_kept"] = \
                    stats.get("owns_changed" if e_["team"] != TEAMS[defend] else "owns_kept", 0) + 1
                e_["team"], e_["reason"], e_["team_source"] = TEAMS[defend], "box_distribution", "behaviour"
        for row, frame in ((s, lo), (o, vf)):
            if not any(e_["event"] == "own" and e_["onset_frame"] == int(D["vf"][row]) for e_ in ev):
                ev.append({"video_frame": frame, "onset_frame": int(D["vf"][row]), "slot": None,
                           "team": TEAMS[defend], "event": "own", "dist_m": np.nan,
                           "team_source": "behaviour", "reason": "box_distribution"})
    return ev, shown, stats


def owns_from_events(D, ev):
    """[(onset_row, team code, reason, video_frame)] for every own event."""
    return [(D["row_of"][e["onset_frame"]], 0 if e["team"] == "H" else 1, e["reason"], e["video_frame"])
            for e in ev if e["event"] == "own" and e["onset_frame"] in D["row_of"]]


def labels(D, owns, events, p):
    """(label 'H'/'A'/'', frame reason, label source) for every graded frame.

    The label source says where the frame's label comes from: "<reason>@<video frame of the own>"
    (or start_of_half@<frame of the first player near the ball>); used by --show and the report.
    """
    n = len(D["vf"])
    lab = np.array([""] * n, dtype=object)
    base = np.array([""] * n, dtype=object)
    src = np.array([""] * n, dtype=object)
    owns = sorted(owns)
    half = D["half"]
    deflect = np.zeros(n, dtype=bool)
    defl_rows = sorted(D["row_of"][e["onset_frame"]] for e in events if e["event"] == "deflect")
    own_rows = [o[0] for o in owns]
    for r in defl_rows:                                   # deflection state until the next own
        k = np.searchsorted(own_rows, r, side="right")
        end = own_rows[k] if k < len(own_rows) else n
        end = min(end, r + np.searchsorted(half[r:], half[r], side="right"))
        deflect[r + 1:end] = True

    for h, first, last in T.graded_blocks():
        rows = np.flatnonzero(half == h)
        if not len(rows):
            continue
        a, b = rows[0], rows[-1] + 1
        hs = [o for o in owns if a <= o[0] < b]
        first_own = hs[0][0] if hs else b
        within = np.flatnonzero(D["nd"][a:first_own] <= p.radius)
        if len(within):
            j = a + within[0]
            lab[a:first_own] = TEAMS[D["team"][j, D["nearest"][j]]]
            base[a:first_own] = "start_of_half"
            src[a:first_own] = "start_of_half@%d" % D["vf"][j]
        for k, (o, team, reason, _) in enumerate(hs):
            end = hs[k + 1][0] if k + 1 < len(hs) else b
            lab[o:end] = TEAMS[team]
            base[o:end] = reason
            src[o:end] = "%s@%d" % (reason, D["vf"][o])

    with np.errstate(invalid="ignore"):
        close = D["d"] <= p.radius
        h_close = (close & (D["team"] == 0)).any(axis=1)
        a_close = (close & (D["team"] == 1)).any(axis=1)
        two_teams = h_close & a_close
        reason = np.where(np.isnan(D["bx"]), "hold_missing",
                 np.where(D["speed"] >= C.FAST_BALL, "hold_fast_ball",
                 np.where(two_teams, "hold_close",
                 np.where(deflect, "hold_deflect",
                 np.where(D["nd"] > p.radius, "hold_no_candidate", base))))).astype(object)
    reason[[o[0] for o in owns]] = [o[2] for o in owns]          # the own frame keeps its reason
    keep = (base == "start_of_half") | (lab == "")
    reason[keep] = base[keep]
    return lab, reason, src


def show(D, lab, reason, src, ev, frames_wanted, around, radius, holds=()):
    """Per-frame rows for local debugging; only when --show is given."""
    touch_at = {int(r.video_frame): r for r in D["touches"].itertuples()}
    onset_at = {int(r.onset_frame): r for r in D["touches"].itertuples()}
    ev_at = {}
    for e in ev:
        ev_at.setdefault(e["onset_frame"], []).append(e)
    for v in frames_wanted:
        if v not in D["row_of"]:
            print("\nvideo %d: not in a graded half" % v)
            continue
        print("\nvideo %d  (half %d, video %s, match %s)" % (v, T.half(v), T.video_time(v),
                                                             T.match_time(v)))
        print("   video  ball x,y (m)    m/s   touch / onset        candidates (slot team source dist)"
              "                label  label set by           frame state  events")
        for x in range(v - around, v + around + 1):
            i = D["row_of"].get(x)
            if i is None:
                continue
            ball = "   not seen   " if np.isnan(D["bx"][i]) else "%5.1f,%5.1f   " % (D["bx"][i], D["by"][i])
            sp = "     -" if np.isnan(D["speed"][i]) else "%6.1f" % D["speed"][i]
            tch = ""
            if x in touch_at:
                tch = "%s (onset %d)" % (touch_at[x].kind, touch_at[x].onset_frame)
            elif x in onset_at:
                tch = "onset of %s" % onset_at[x].kind
            within = np.flatnonzero(D["d"][i] <= radius)
            cands = " ".join("%d%s%s %.1f" % (s + 1, TEAMS[D["team"][i, s]],
                                              "k" if D["from_keeper"][i, s] else "",
                                              D["d"][i, s]) for s in within)
            evs = " ".join("%s:%d" % (e["event"], e["slot"]) for e in ev_at.get(x, []))
            state = reason[i] if reason[i].startswith("hold_") else ""
            print("  %6d  %s %s  %-20s %-38s %-5s %-22s %-17s %s%s" % (
                x, ball, sp, tch, cands, lab[i] or "-", src[i] or "-", state, evs,
                "  <" if x == v else ""))
            # always: nearest usable H and A player (even outside the radius) and the keepers
            near = []
            for code in (0, 1):
                dd = np.where(D["team"][i] == code, D["d"][i], np.inf)
                s = int(np.argmin(dd))
                near.append("nearest %s: %s" % (TEAMS[code], "-" if not np.isfinite(dd[s]) else
                            "slot %d%s %.1f m" % (s + 1, "k" if D["from_keeper"][i, s] else "", dd[s])))
            kp = D["kp"].iloc[i]
            keep = "keepers: H %s (%s), A %s (%s)" % (
                "-" if pd.isna(kp["h_keeper_slot"]) else int(kp["h_keeper_slot"]), kp["h_source"],
                "-" if pd.isna(kp["a_keeper_slot"]) else int(kp["a_keeper_slot"]), kp["a_source"])
            print("          %s; %s; %s" % (near[0], near[1], keep))
            for hd in holds:
                if hd["start"] <= i <= max(hd["end"], hd.get("onset", hd["end"])):
                    print("          box hold from video %d (%s penalty area, defended by %s): %s%s" % (
                        D["vf"][hd["start"]], "left" if hd["side"] == 0 else "right", TEAMS[hd["team"]],
                        "" if "travel" not in hd else
                        "flight to video %d (%s): %.1f m, gain %.1f m; " % (
                            D["vf"][hd["flight_end"]], hd["flight_how"], hd["travel"], hd["gain"]),
                        "FIRED (box_distribution)" if hd["fired"] else "not fired: " + hd["why"]))


def main():
    out = os.path.join(REPO, "out")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=os.path.join(out, "clean_table.csv"))
    ap.add_argument("--touches", default=os.path.join(out, "touches.csv"))
    ap.add_argument("--trips", default=os.path.join(out, "ball_trips.csv"))
    ap.add_argument("--keepers", default=os.path.join(out, "keepers.csv"))
    ap.add_argument("--out", default=os.path.join(out, "labels_raw.csv"))
    ap.add_argument("--events-out", default=os.path.join(out, "events.csv"))
    ap.add_argument("--radius", type=float, default=C.CONTROL_RADIUS,
                    help="m, CONTROL_RADIUS: close events, 50/50, gap reappearance, start of half")
    ap.add_argument("--attrib-radius", type=float, default=C.TOUCH_ATTRIB_RADIUS,
                    help="m, TOUCH_ATTRIB_RADIUS: finding the toucher and the own check")
    ap.add_argument("--min-own", type=int, default=C.MIN_OWN_FRAMES)
    ap.add_argument("--show", type=int, nargs="+", metavar="VF",
                    help="print per-frame rows around these video frames (match data)")
    ap.add_argument("--around", type=int, default=25)
    p = ap.parse_args()
    hints = {"table": "Run step 1: python src/clean_table.py",
             "touches": "Run step 4: python src/ball_trips.py",
             "trips": "Run step 4: python src/ball_trips.py",
             "keepers": "Run step 3: python src/team_check.py"}
    for name, hint in hints.items():
        if not os.path.isfile(getattr(p, name)):
            fail_missing(name, getattr(p, name), hint)

    t0 = time.time()
    D = load(p)
    ev, owns, tstats = touch_events(D, p)
    gev, gowns = gap_events(D, p, [D["row_of"][v] for v in D["touches"]["onset_frame"].tolist()
                                   if v in D["row_of"]])
    ev += gev
    ev, holds, bstats = box_distribution(D, ev, p, home_defends_first_half(
        games.asset_paths(source="tactical")["atd_meta"]))
    owns = owns_from_events(D, ev)
    ev.sort(key=lambda e: (e["onset_frame"], e["video_frame"], e["event"],
                           -1 if e["slot"] is None else e["slot"]))
    lab, reason, src = labels(D, owns, ev, p)

    for path in (p.out, p.events_out):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    edf = pd.DataFrame(ev, columns=EVENT_COLUMNS)
    edf["slot"] = edf["slot"].astype("Int64")
    edf.to_csv(p.events_out, index=False, float_format="%.3f", lineterminator="\n")
    ldf = pd.DataFrame({"video_frame": D["vf"], "label": lab, "reason": reason})
    ldf.to_csv(p.out, index=False, lineterminator="\n")

    # -- aggregates only ------------------------------------------------------------------------
    print("wrote %d labels to %s and %d events to %s" % (len(ldf), p.out, len(edf), p.events_out))
    changes_total = 0
    spells = []
    for h in sorted(set(D["half"].tolist())):
        m = D["half"] == h
        l = lab[m]
        known = l[l != ""]
        ch = int((known[1:] != known[:-1]).sum())
        changes_total += ch
        edges = np.flatnonzero(np.concatenate(([True], l[1:] != l[:-1], [True])))
        for s, e in zip(edges[:-1], edges[1:]):
            if l[s] in ("H", "A"):
                spells.append((e - s) / C.FPS)
        print("half %d: H %.1f%%  A %.1f%%  empty %.1f%%  possession changes %d" % (
            h, 100 * (l == "H").mean(), 100 * (l == "A").mean(), 100 * (l == "").mean(), ch))
    sp = np.array(spells)
    print("match: possession changes %d; spells %d, length s p10 %.1f p25 %.1f p50 %.1f p75 %.1f "
          "p90 %.1f max %.0f" % (changes_total, len(sp), *np.percentile(sp, [10, 25, 50, 75, 90]),
                                 sp.max()))
    print("events: %s" % ", ".join("%s %d" % kv for kv in edf["event"].value_counts().sort_index().items()))
    o = edf[edf["event"] == "own"]
    print("owns by reason: %s" % ", ".join("%s %d" % kv for kv in o["reason"].value_counts().sort_index().items()))
    print("owns with team_source = keeper: %.1f%% (%d of %d)" % (
        100 * (o["team_source"] == "keeper").mean(), int((o["team_source"] == "keeper").sum()), len(o)))
    n_t = sum(tstats["touches"].values())
    print("touches: %d after dropping %d with the same onset as a higher-priority touch; no toucher "
          "within %.1f m at onset +-%d frames: %.1f%% (%d)" % (
              n_t, tstats["same_onset_dropped"], p.attrib_radius, C.TOUCH_ATTRIB_FRAMES,
              100 * tstats["no_toucher"] / max(1, n_t), tstats["no_toucher"]))
    print("attributed by kind: %s" % ", ".join(
        "%s %.1f%% (%d/%d)" % (k, 100 * tstats["attributed"].get(k, 0) / v,
                               tstats["attributed"].get(k, 0), v)
        for k, v in sorted(tstats["touches"].items())))
    print("own_touch owns with ball >= FAST_BALL during the own check: %d" % tstats["fast_own_touch"])
    for h, first, _ in T.graded_blocks():
        rows = [o[0] for o in owns if D["half"][o[0]] == h]
        if rows:
            print("half %d: first own %.1f s after the first graded frame (%s)" % (
                h, (D["vf"][min(rows)] - first) / C.FPS, src[min(rows)]))
    print("box_distribution: fired %s; on the distributing touch the toucher's slot team was changed "
          "%d, already right %d, no toucher %d; toucher is keepers.csv's keeper of that goal %d, "
          "not %d; holds skipped inside a ball-out stretch %d; penalty-area holds found %d" % (
              ", ".join("half %d %s-goal %d" % (h, t, c) for (h, t), c in sorted(bstats["fired"].items()))
              or "0", bstats["changed"], bstats["unchanged"], bstats["no_toucher"],
              bstats["keeper_agree"], bstats["keeper_disagree"], bstats["skipped_dead"], len(holds)))
    print("  owns inside the fired holds: team changed %d, already right %d" % (
        bstats.get("owns_changed", 0), bstats.get("owns_kept", 0)))
    why = {}
    for hd in holds:
        if not hd["fired"]:
            k = hd["why"].split(":")[0] if hd["why"].startswith("trip") else hd["why"]
            k = "trip too short or not away from goal" if k.startswith("trip") else k
            why[k] = why.get(k, 0) + 1
    print("  holds not fired: %s" % (", ".join("%s %d" % kv for kv in sorted(why.items())) or "-"))
    ends = {}
    for hd in holds:
        if "flight_how" in hd:
            ends[hd["flight_how"]] = ends.get(hd["flight_how"], 0) + 1
    print("  measured flights ended at: %s" % (", ".join("%s %d" % kv for kv in sorted(ends.items())) or "-"))
    print("gap_reappear owns: %d (of %d trips starting after a long gap)" % (
        int((o["reason"] == "gap_reappear").sum()), int((D["trips"]["starts_with"] == "gap").sum())))
    print("frame reasons: %s" % ", ".join("%s %.1f%%" % (k, 100 * v / len(ldf)) for k, v in
                                          ldf["reason"].value_counts().sort_index().items()))
    print("run time %.1f s" % (time.time() - t0))

    if p.show:
        print("\n--show: per-frame rows for local debugging")
        show(D, lab, reason, src, ev, p.show, p.around, p.radius, holds)


if __name__ == "__main__":
    main()
