"""Step 4: clean the ball path, find touches, split it into trips between touches.

Input: out/clean_table.csv (step 1). Thresholds from src/config.py; CLI flags override them.

1. Clean the path, per half:
   - glitch mask: a frame whose step-1 ball speed is above SANE_BALL_SPEED is set missing. One
     bad point gives two spikes (the jump out and the jump back), so the good frame after it is
     masked too -- one frame lost, never a fake touch.
   - smoothing: centred SMOOTH_FRAMES mean where all frames are present (never bridges a gap).
     Off by default (SMOOTH_FRAMES = 1): the tracker's ball is already smooth and interpolated.
   - speed[i] = the step from frame i-1 to i (needs both present).

2. Touches. The tracker's ball is smoothed, so a kick or trap shows as a speed ramp over about a
   quarter of a second (TOUCH_WINDOW_FRAMES = 6), not as a jump between two frames.
   - speed change: flag frame i when speed[i + TOUCH_WINDOW_FRAMES] - speed[i] reaches
     +TOUCH_SPEED_CHANGE (speed_up, a kick) or -TOUCH_SPEED_CHANGE (speed_down, a trap or save).
     Each run of flags is one ramp; its touch goes on the frame of the largest one-frame speed
     change inside the ramp.
   - turn: angle between the chords p[i-K] -> p[i] and p[i] -> p[i+K] (K = TURN_CHORD_FRAMES, all
     frames present) >= TOUCH_ANGLE, only if both chords move faster than TOUCH_MIN_MOVING_SPEED.
     Each run's touch goes on its largest angle.
   - candidates closer than TOUCH_MIN_GAP_FRAMES (chained, >= the window) merge into one touch:
     the strongest speed change wins, else the largest turn. speed_before / speed_after are the
     speeds half a window before / after the touch frame.
   - stop_go: a ball on grass only slows down by itself. A local speed minimum below STOP_SPEED
     followed within STOP_GO_FRAMES by a rise >= STOP_GO_RISE is a touch at the minimum; the heading
     change across the slowdown is stored in turn_deg.
   - merge priority: stop_go, then the strongest speed change, then the largest turn.
   - onset_frame: where step 5 identifies the toucher -- speed_up: the last local speed minimum
     before the touch (the kick began there), at most ONSET_MAX_FRAMES back; speed_down: the end of
     the slowdown ramp, the first frame after the touch where the ball stops slowing, at most
     ONSET_MAX_FRAMES ahead (the smoothing starts the slowdown before contact; the receiver has the
     ball at his feet when it stops slowing); turn / stop_go: the touch frame.

3. Trips: the ball path between touches. A trip never spans a half boundary or a missing-ball gap
   longer than MAX_GAP_FRAMES; such a gap ends the trip with no end touch. kind:
   - kick:    the start touch is a speed_up (the speed jumps)
   - throw:   no jump, but the speed rises from below TOUCH_MIN_MOVING_SPEED to >= FAST_BALL over
              >= THROW_RISE_FRAMES with no gap (the keeper-throw pattern)
   - dribble: slow (peak < FAST_BALL), between two touches, at most DRIBBLE_MAX_TRIP_FRAMES long
   - unknown: anything else

Outputs: out/ball_trips.csv, out/touches.csv. Stdout: aggregates only.

    python src/ball_trips.py
    python src/ball_trips.py --show <vf> ... --around 25     # per-frame rows for local debugging
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
    sys.exit("Can't find games.py. Run this from the repo root: python src/ball_trips.py")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import timeutil as T     # noqa: E402
import config as C       # noqa: E402

TRIP_COLUMNS = ["trip_id", "half", "start_frame", "end_frame", "start_speed", "peak_speed",
                "end_speed", "kind", "starts_with", "ends_with"]
TOUCH_COLUMNS = ["video_frame", "half", "kind", "speed_before", "speed_after", "onset_frame",
                 "turn_deg"]


def runs(mask):
    """[(start, stop)] half-open runs of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def clean_path(x, y, raw_speed, p):
    """(present, xs, ys, speed, heading, n_masked) for one half; arrays indexed by frame offset."""
    present = ~np.isnan(x)
    glitch = present & (raw_speed > p.sane)
    present &= ~glitch
    xs, ys = np.where(present, x, np.nan), np.where(present, y, np.nan)
    k = p.smooth // 2
    if k:
        n = len(xs)
        full = np.zeros(n, dtype=bool)
        sx, sy = np.zeros(n), np.zeros(n)
        full[k:n - k] = True
        for d in range(-k, k + 1):
            full[k:n - k] &= present[k + d:n - k + d]
            sx[k:n - k] += np.nan_to_num(xs[k + d:n - k + d])
            sy[k:n - k] += np.nan_to_num(ys[k + d:n - k + d])
        xs = np.where(full, sx / p.smooth, xs)
        ys = np.where(full, sy / p.smooth, ys)
    dx = np.full(len(xs), np.nan)
    dy = np.full(len(xs), np.nan)
    dx[1:], dy[1:] = xs[1:] - xs[:-1], ys[1:] - ys[:-1]
    speed = np.hypot(dx, dy) * C.FPS                       # NaN unless both frames present
    heading = np.arctan2(dy, dx)
    return present, xs, ys, speed, heading, int(glitch.sum())


def peak_in(values, a, b, sign):
    """Index in [a, b) of the largest sign*value (NaN never wins), or None if all NaN."""
    seg = sign * values[a:b]
    if not len(seg) or np.isnan(seg).all():
        return None
    return a + int(np.nanargmax(seg))


def stop_go_candidates(speed, heading, p):
    """[(m, "stop_go", rise, heading change deg)]: the ball stopped and then moved again.

    A ball on grass only slows down by itself. So a local speed minimum below STOP_SPEED followed
    within STOP_GO_FRAMES by a rise of at least STOP_GO_RISE is a touch, placed at the minimum.
    The heading change compares the last step faster than TOUCH_MIN_MOVING_SPEED before the minimum
    (within ONSET_MAX_FRAMES) with the first one after it (within STOP_GO_FRAMES); NaN if missing.
    """
    n, N = len(speed), C.STOP_GO_FRAMES
    with np.errstate(invalid="ignore"):
        low = np.zeros(n, dtype=bool)
        low[1:-1] = (speed[1:-1] <= speed[:-2]) & (speed[1:-1] <= speed[2:]) & \
                    (speed[1:-1] < C.STOP_SPEED)
        ahead = np.full(n, np.nan)
        for d in range(1, N + 1):
            ahead[:n - d] = np.fmax(ahead[:n - d], speed[d:])
        rise = ahead - speed
        hit = low & (rise >= C.STOP_GO_RISE)
    out = []
    for a, b in runs(hit):
        m = a + int(np.nanargmin(speed[a:b]))
        before = [k for k in range(m, max(0, m - C.ONSET_MAX_FRAMES) - 1, -1)
                  if speed[k] >= p.min_moving]
        after = [k for k in range(m + 1, min(n, m + N + 1)) if speed[k] >= p.min_moving]
        ang = np.nan
        if before and after:
            d = abs(heading[after[0]] - heading[before[0]])
            ang = float(np.degrees(min(d, 2 * np.pi - d)))
        out.append((m, "stop_go", float(rise[m]), ang))
    return out


def onset_of(f, kind, speed):
    """Frame where the toucher is identified.

    speed_up: the last local speed minimum before f (the kick began there), at most ONSET_MAX_FRAMES
    back. speed_down: the end of the slowdown ramp -- the first frame after f where the ball stops
    slowing, at most ONSET_MAX_FRAMES ahead: the tracker's smoothing starts the slowdown before the
    contact, and the receiver has the ball at his feet when it stops slowing. turn / stop_go: f.
    """
    k = f
    if kind == "speed_up":
        lo = max(0, f - C.ONSET_MAX_FRAMES)
        while k - 1 >= lo and not np.isnan(speed[k - 1]) and speed[k - 1] <= speed[k]:
            k -= 1
    elif kind == "speed_down":
        hi = min(len(speed) - 1, f + C.ONSET_MAX_FRAMES)
        while k + 1 <= hi and not np.isnan(speed[k + 1]) and speed[k + 1] < speed[k]:
            k += 1
    return k


PRIORITY = {"stop_go": 0, "speed_up": 1, "speed_down": 1, "turn": 2}


def find_touches(speed, heading, xs, ys, present, p):
    """[(offset, kind, speed_before, speed_after, onset, turn_deg)] for one half, plus dv and turn.

    Speed: flag frames where speed[i + W] - speed[i] reaches +-touch_speed (W = window); each run of
    flags is one ramp, and its touch goes on the frame of the largest one-frame speed change inside
    the ramp. Turn: angle between the chords p[i-K]->p[i] and p[i]->p[i+K] (K = chord, all frames
    present, both chords moving); each flagged run's touch goes on its largest angle. Candidates
    closer than min_gap merge into one touch: a stop_go wins, else the strongest speed change, else
    the largest turn. onset = onset_of(); turn_deg = the angle for turn and stop_go touches.
    """
    n = len(speed)
    W, K = p.window, p.chord
    dv = np.full(n, np.nan)
    dv[:n - W] = speed[W:] - speed[:n - W]
    acc = np.full(n, np.nan)                       # change happening at frame j: step j -> j+1
    acc[:-1] = speed[1:] - speed[:-1]

    turn = np.full(n, np.nan)
    if n > 2 * K:
        miss = np.concatenate(([0], np.cumsum(~present)))
        i = np.arange(K, n - K)
        whole = (miss[i + K + 1] - miss[i - K]) == 0                  # p[i-K .. i+K] all present
        ix, iy = xs[i] - xs[i - K], ys[i] - ys[i - K]
        ox, oy = xs[i + K] - xs[i], ys[i + K] - ys[i]
        moving = (np.hypot(ix, iy) * C.FPS / K >= p.min_moving) & \
                 (np.hypot(ox, oy) * C.FPS / K >= p.min_moving)
        d = np.abs(np.arctan2(oy, ox) - np.arctan2(iy, ix))
        ang = np.degrees(np.minimum(d, 2 * np.pi - d))
        turn[i] = np.where(whole & moving, ang, np.nan)

    with np.errstate(invalid="ignore"):
        up, down, turning = dv >= p.touch_speed, dv <= -p.touch_speed, turn >= p.touch_angle
    cand = []
    for mask, kind, sign in ((up, "speed_up", 1), (down, "speed_down", -1)):
        for a, b in runs(mask):
            j = peak_in(acc, a, min(n - 1, b - 1 + W), sign)     # ramp spans a .. b-1+W
            if j is not None:
                cand.append((j, kind, abs(acc[j]), np.nan))
    for a, b in runs(turning):
        j = peak_in(turn, a, b, 1)
        cand.append((j, "turn", turn[j], turn[j]))
    cand += stop_go_candidates(speed, heading, p)
    cand.sort(key=lambda c: (c[0], c[1]))

    def at(k):
        return speed[k] if 0 <= k < n else np.nan

    touches, group = [], []

    def close_group():
        top = min(PRIORITY[c[1]] for c in group)
        best = max((c for c in group if PRIORITY[c[1]] == top), key=lambda c: (c[2], -c[0]))
        f, kind = best[0], best[1]
        touches.append((f, kind, at(f - W // 2), at(f + W // 2), onset_of(f, kind, speed), best[3]))

    for c in cand:
        if group and c[0] - group[-1][0] >= p.min_gap:
            close_group()
            group = []
        group.append(c)
    if group:
        close_group()
    touches.sort(key=lambda t: t[0])
    return touches, dv, turn


def build_trips(half, vf0, present, speed, touches, p, start_id):
    """Trips for one half. Touch offsets split trips; gaps > max_gap end them."""
    n = len(present)
    seen = np.flatnonzero(present)
    if not len(seen):
        return []
    # track segments: present frames, joined across gaps of <= max_gap missing frames
    segs, s0 = [], seen[0]
    for prev, cur in zip(seen[:-1], seen[1:]):
        if cur - prev - 1 > p.max_gap:
            segs.append((s0, prev))
            s0 = cur
    segs.append((s0, seen[-1]))

    touch_at = {t[0]: t[1] for t in touches}
    trips = []
    for si, (a, b) in enumerate(segs):
        cuts = sorted(t for t in touch_at if a <= t <= b)
        starts = ([a] if not cuts or cuts[0] != a else []) + cuts
        for k, st in enumerate(starts):
            en = (starts[k + 1] - 1) if k + 1 < len(starts) else b
            if en < st:
                continue
            starts_with = "touch" if st in touch_at else ("half_start" if si == 0 else "gap")
            if k + 1 < len(starts):
                ends_with = "touch"
            else:
                ends_with = "half_end" if si == len(segs) - 1 else "gap"
            sp = speed[st + 1:en + 1]
            ok = ~np.isnan(sp)
            vals = sp[ok]
            start_speed = vals[0] if len(vals) else np.nan
            end_speed = vals[-1] if len(vals) else np.nan
            peak = vals.max() if len(vals) else np.nan
            kind = trip_kind(st, en, sp, peak, touch_at.get(st), starts_with, ends_with, p)
            trips.append([start_id + len(trips), half, vf0 + st, vf0 + en, start_speed, peak,
                          end_speed, kind, starts_with, ends_with])
    return trips


def trip_kind(st, en, sp, peak, start_kind, starts_with, ends_with, p):
    if start_kind == "speed_up":
        return "kick"
    if not np.isnan(peak) and peak >= p.fast:
        pk = int(np.nanargmax(sp))
        before = sp[:pk]
        slow = np.flatnonzero(before < p.min_moving)
        if len(slow):
            rise = sp[slow[-1]:pk + 1]
            if pk - slow[-1] >= C.THROW_RISE_FRAMES and not np.isnan(rise).any():
                return "throw"
    if (starts_with == "touch" and ends_with == "touch" and en - st + 1 <= C.DRIBBLE_MAX_TRIP_FRAMES
            and (np.isnan(peak) or peak < p.fast)):
        return "dribble"
    return "unknown"


def show(per_half, frames_wanted, around):
    """Per-frame rows for local debugging; only when --show is given."""
    for v in frames_wanted:
        h = T.half(v)
        if h is None or h not in per_half:
            print("\nvideo %d: not in a graded half" % v)
            continue
        r = per_half[h]
        print("\nvideo %d  (half %d, video %s, match %s)" % (v, h, T.video_time(v), T.match_time(v)))
        print("   video  seen  x_m     y_m     speed   dv      turn    touch        trip")
        for x in range(v - around, v + around + 1):
            if T.half(x) != h:
                continue
            i = x - r["vf0"]
            f = lambda a, w=7: ("%*.2f" % (w, a)) if not np.isnan(a) else " " * (w - 1) + "-"
            touch = r["touch_kind"].get(i, "") or (
                "onset " + r["onset_of"][i] if i in r["onset_of"] else "")
            trip = r["trip_of"][i]
            print("  %6d  %-4s  %s %s %s %s %s  %-11s  %s%s" % (
                x, "yes" if r["present"][i] else ("mask" if r["masked"][i] else "no"),
                f(r["xs"][i]), f(r["ys"][i]), f(r["speed"][i]), f(r["dv"][i]), f(r["turn"][i]),
                touch, trip, "  <" if x == v else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=os.path.join(REPO, "out", "clean_table.csv"))
    ap.add_argument("--out", default=os.path.join(REPO, "out", "ball_trips.csv"))
    ap.add_argument("--touches-out", default=os.path.join(REPO, "out", "touches.csv"))
    ap.add_argument("--touch-speed", type=float, default=C.TOUCH_SPEED_CHANGE)
    ap.add_argument("--touch-angle", type=float, default=C.TOUCH_ANGLE)
    ap.add_argument("--window", type=int, default=C.TOUCH_WINDOW_FRAMES,
                    help="frames over which a speed change is measured")
    ap.add_argument("--chord", type=int, default=C.TURN_CHORD_FRAMES,
                    help="chord length in frames for direction changes")
    ap.add_argument("--fast", type=float, default=C.FAST_BALL)
    ap.add_argument("--min-moving", type=float, default=C.TOUCH_MIN_MOVING_SPEED)
    ap.add_argument("--min-gap", type=int, default=C.TOUCH_MIN_GAP_FRAMES)
    ap.add_argument("--max-gap", type=int, default=C.MAX_GAP_FRAMES)
    ap.add_argument("--smooth", type=int, default=C.SMOOTH_FRAMES, help="odd; 1 = off")
    ap.add_argument("--sane", type=float, default=C.SANE_BALL_SPEED)
    ap.add_argument("--show", type=int, nargs="+", metavar="VF",
                    help="print per-frame ball rows around these video frames (match data)")
    ap.add_argument("--around", type=int, default=25)
    p = ap.parse_args()
    if not os.path.isfile(p.table):
        sys.exit("Can't find the table file (%s). Run step 1 first: python src/clean_table.py"
                 % p.table)
    if p.smooth < 1 or p.smooth % 2 == 0:
        sys.exit("--smooth must be an odd number >= 1")

    t0 = time.time()
    table = pd.read_csv(p.table, usecols=["video_frame", "half", "ball_x_m", "ball_y_m",
                                          "ball_speed"])
    trips, touch_rows, per_half, stats = [], [], {}, {}
    for h, part in table.groupby("half", sort=True):
        vf = part["video_frame"].to_numpy()
        if not np.array_equal(vf, np.arange(vf[0], vf[0] + len(vf))):
            sys.exit("clean_table half %d is not contiguous; rerun step 1 for the full half" % h)
        present, xs, ys, speed, heading, masked = clean_path(
            part["ball_x_m"].to_numpy(), part["ball_y_m"].to_numpy(),
            part["ball_speed"].to_numpy(), p)
        touches, dv, turn = find_touches(speed, heading, xs, ys, present, p)
        trips_h = build_trips(h, vf[0], present, speed, touches, p, len(trips) + 1)
        trips += trips_h
        touch_rows += [[vf[0] + i, h, k, b, a, vf[0] + o, g] for i, k, b, a, o, g in touches]
        stats[h] = {"masked": masked, "seen_min": present.sum() / C.FPS / 60.0,
                    "touches": len(touches), "trips": len(trips_h)}
        if p.show:
            trip_of = np.array([""] * len(vf), dtype=object)
            for t in trips_h:
                trip_of[t[2] - vf[0]:t[3] - vf[0] + 1] = "%d %s" % (t[0], t[7])
            per_half[h] = {"vf0": vf[0], "present": present, "masked": (~np.isnan(
                part["ball_x_m"].to_numpy())) & ~present, "xs": xs, "ys": ys, "speed": speed,
                "dv": dv, "turn": turn, "touch_kind": {t[0]: t[1] for t in touches},
                "onset_of": {t[4]: t[1] for t in touches if t[4] != t[0]},
                "trip_of": trip_of}

    for path in (p.out, p.touches_out):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tdf = pd.DataFrame(trips, columns=TRIP_COLUMNS)
    tdf.to_csv(p.out, index=False, float_format="%.3f", lineterminator="\n")
    xdf = pd.DataFrame(touch_rows, columns=TOUCH_COLUMNS)
    xdf.to_csv(p.touches_out, index=False, float_format="%.3f", lineterminator="\n")

    # -- aggregates only ------------------------------------------------------------------------
    print("wrote %d trips to %s and %d touches to %s" % (len(tdf), p.out, len(xdf), p.touches_out))
    for h in sorted(stats):
        s = stats[h]
        print("half %d: masked glitch frames %d; ball seen %.1f min; touches %d (%.1f per seen "
              "minute); trips %d" % (h, s["masked"], s["seen_min"], s["touches"],
                                     s["touches"] / s["seen_min"], s["trips"]))
    total_min = sum(s["seen_min"] for s in stats.values())
    print("match: touches %d (%.1f per seen minute)" % (len(xdf), len(xdf) / total_min))
    print("touch kinds: %s" % ", ".join("%s %d" % kv for kv in
                                         xdf["kind"].value_counts().sort_index().items()))
    off = (xdf["onset_frame"] - xdf["video_frame"]).to_numpy()
    print("onset more than 5 frames before the placed frame: %.1f%% of all touches" % (
        100 * (off < -5).mean()))
    for kind, sign, word in (("speed_up", -1, "look-back"), ("speed_down", 1, "look-ahead")):
        v = sign * off[(xdf["kind"] == kind).to_numpy()]
        if len(v):
            print("  %s onset %s frames: p10 %d p25 %d p50 %d p75 %d p90 %d max %d (limit %d)" % (
                kind, word, *np.percentile(v, [10, 25, 50, 75, 90]).astype(int), int(v.max()),
                C.ONSET_MAX_FRAMES))
    sg = xdf[xdf["kind"] == "stop_go"]["turn_deg"]
    if len(sg):
        print("stop_go heading change deg: p25 %.0f p50 %.0f p75 %.0f (%d of %d with both headings)" % (
            *np.nanpercentile(sg, [25, 50, 75]), int(sg.notna().sum()), len(sg)))
    print("trip kinds:  %s" % ", ".join("%s %d" % kv for kv in
                                         tdf["kind"].value_counts().sort_index().items()))
    dur = (tdf["end_frame"] - tdf["start_frame"] + 1).to_numpy() / C.FPS
    print("trip duration s: p10 %.2f  p25 %.2f  p50 %.2f  p75 %.2f  p90 %.2f  max %.1f" % tuple(
        np.percentile(dur, [10, 25, 50, 75, 90]).tolist() + [dur.max()]))
    print("trips ending at a long gap: %.1f%%   (ending at a touch %.1f%%, at half end %.1f%%)" % (
        100 * (tdf["ends_with"] == "gap").mean(), 100 * (tdf["ends_with"] == "touch").mean(),
        100 * (tdf["ends_with"] == "half_end").mean()))
    print("run time %.1f s" % (time.time() - t0))

    if p.show:
        print("\n--show: per-frame rows for local debugging")
        show(per_half, p.show, p.around)


if __name__ == "__main__":
    main()
