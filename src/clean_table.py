"""Step 1: one clean row per graded video frame, from the raw tactical tracking.

Columns:

    video_frame, half, video_time, match_time, ball_x_m, ball_y_m, ball_speed,
    near_h_slot, near_h_dist, near_a_slot, near_a_dist

- Graded frames only, half bounds from games.half_video_blocks().
- ball_speed = distance between this frame's and the previous frame's ball x FPS (m/s); empty if
  either ball is missing. The ATD's own ball speed field is always NaN.
- Nearest home / away slot to the ball and its distance in metres. Team of a slot comes from the
  metadata XML (starter/baseline.team_of_each_track). Empty slots and frozen slots (see
  frozen_mask: stuck boxes, and drifting abandoned boxes with a live twin beside them) are skipped.
- Empty cells mean "no value" (pandas reads them back as NaN).

Usage (from the repo root):
    python src/clean_table.py                        # both halves -> out/clean_table.csv
    python src/clean_table.py --half 1
    python src/clean_table.py --start <first vf> --end <last vf> --out out/<name>.csv

Prints aggregates only (counts, shares, percentiles), never per-frame rows.
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
    sys.exit("Can't find games.py. Run this from the repo root: python src/clean_table.py")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import games                                        # noqa: E402
from starter.baseline import team_of_each_track, HOME   # noqa: E402
from starter.load_atd import video_frame_of         # noqa: E402
import timeutil as T                                # noqa: E402
import config as C                                  # noqa: E402

COLUMNS = ["video_frame", "half", "video_time", "match_time", "ball_x_m", "ball_y_m",
           "ball_speed", "near_h_slot", "near_h_dist", "near_a_slot", "near_a_dist"]


def fail_missing(name, path):
    sys.exit("Can't find the %s file (%s). Put this script next to games.py, or pass "
             "--%s /path/to/file." % (name, path, name))


def selected_frames(half, start, end):
    """Sorted video frames to output: graded halves, optionally narrowed by --half / --start / --end."""
    out = []
    for h, first, last in T.graded_blocks():
        if half is not None and h != half:
            continue
        lo = first if start is None else max(first, start)
        hi = last if end is None else min(last, end)
        if lo <= hi:
            out.append(np.arange(lo, hi + 1))
    return np.concatenate(out) if out else np.array([], dtype=np.int64)


def read_tracking(atd_path, wanted, n_slots):
    """Parse only the lines whose video frame is in `wanted` (a boolean mask over video frames).

    Returns (vf, players[n, slots, 2] metres, ball[n, 2] metres, n_malformed). A line that does
    not split into three parts or has the wrong number of slot groups (e.g. a cut-off last line)
    is skipped and counted.
    """
    vfs, players, balls = [], [], []
    malformed = 0
    with open(atd_path) as handle:
        for line in handle:
            head, sep, rest = line.partition(":")
            try:
                vf = video_frame_of(head)
            except ValueError:
                malformed += 1
                continue
            if vf < 0 or vf >= len(wanted) or not wanted[vf]:
                continue
            parts = rest.strip().split(":")
            try:
                if len(parts) != 2:
                    raise ValueError
                slot_vals = np.array(parts[0].replace(";", ",").split(","), dtype=np.float64)
                if slot_vals.size != n_slots * 3:
                    raise ValueError
                ball_vals = (parts[1].split(",") + ["NaN", "NaN"])[:2]
                ball = np.array(ball_vals, dtype=np.float64)
            except ValueError:
                malformed += 1
                continue
            vfs.append(vf)
            players.append(slot_vals.reshape(n_slots, 3)[:, :2])
            balls.append(ball)
    vf = np.array(vfs, dtype=np.int64)
    order = np.argsort(vf, kind="stable")
    scale = np.array([C.PITCH_L, C.PITCH_W])
    players = (np.array(players).reshape(-1, n_slots, 2) * scale)[order]
    balls = (np.array(balls).reshape(-1, 2) * scale)[order]
    return vf[order], players, balls, malformed


def _windows(vf, players, min_frames, max_path):
    """([n, slots] in a window of min_frames present frames with path == 0,
        [n, slots] in such a window with path < max_path)."""
    n, slots, _ = players.shape
    zero = np.zeros((n, slots), dtype=bool)
    still = np.zeros((n, slots), dtype=bool)
    if n < min_frames:
        return zero, still
    consecutive = np.concatenate(([False], np.diff(vf) == 1))
    big = 1e9                                               # breaks a window: gap or missing frame
    j = np.arange(n - min_frames + 1)
    for s in range(slots):
        x, y = players[:, s, 0], players[:, s, 1]
        step = np.full(n, big)
        step[1:] = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])
        step = np.where(consecutive & ~np.isnan(step), step, big)
        cum = np.concatenate(([0.0], np.cumsum(step)))
        # window of frames j .. j+min_frames-1 has steps j+1 .. j+min_frames-1; a gap adds `big`
        path = cum[j + min_frames] - cum[j + 1]
        valid = ~np.isnan(x[j])
        for out, ok in ((zero, valid & (path == 0.0)), (still, valid & (path < max_path))):
            mark = np.zeros(n + 1, dtype=np.int64)
            np.add.at(mark, j[ok], 1)
            np.add.at(mark, j[ok] + min_frames, -1)
            out[:, s] = np.cumsum(mark[:n]) > 0
    return zero, still


def frozen_mask(vf, players, min_frames, max_path=None, twin_radius=None, details=False):
    """[n, slots] True where a slot is a stuck or abandoned (ghost) box.

    Frozen if (a) its position is bit-identical over min_frames consecutive frames, or (b) it is in a
    still stretch (total path over min_frames frames below max_path = FROZEN_MAX_PATH_M) that is a
    ghost: at any frame of the first min_frames frames of the stretch, a live box (present, not
    still) of another slot was within twin_radius (FROZEN_TWIN_RADIUS_M). The twin is checked when
    the stretch STARTS, because the person walks away from his abandoned box before it has been still
    for min_frames; once a ghost, the stretch stays frozen until the box moves again. A lone still
    box (a keeper standing still while play is far away) is not a ghost.

    details=True also returns (still, ghost) masks for diagnostics.
    """
    max_path = C.FROZEN_MAX_PATH_M if max_path is None else max_path
    twin_radius = C.FROZEN_TWIN_RADIUS_M if twin_radius is None else twin_radius
    zero, still = _windows(vf, players, min_frames, max_path)
    present = ~np.isnan(players[:, :, 0])
    live = present & ~still
    consecutive = np.concatenate(([False], np.diff(vf) == 1))
    ghost = np.zeros_like(still)
    n, slots = still.shape
    for s in range(slots):
        col = still[:, s]
        if not col.any():
            continue
        starts = np.flatnonzero(col & ~np.concatenate(([False], col[:-1] & consecutive[1:])))
        for a in starts:
            b = a
            while b + 1 < n and col[b + 1] and consecutive[b + 1]:
                b += 1
            head = slice(a, min(b + 1, a + min_frames))
            d = np.hypot(players[head, :, 0] - players[head, s, 0][:, None],
                         players[head, :, 1] - players[head, s, 1][:, None])
            near = (d <= twin_radius) & live[head]
            near[:, s] = False
            if near.any():
                ghost[a:b + 1, s] = True
    frozen = zero | ghost
    return (frozen, still, ghost) if details else frozen


def nearest(players, ball, usable, team_slots):
    """(slot number 1-based or -1, distance m or NaN) of the nearest usable slot in team_slots."""
    p = players[:, team_slots, :]
    d = np.hypot(p[:, :, 0] - ball[:, None, 0], p[:, :, 1] - ball[:, None, 1])
    d = np.where(usable[:, team_slots] & ~np.isnan(d), d, np.inf)
    idx = np.argmin(d, axis=1)
    best = d[np.arange(len(d)), idx]
    found = np.isfinite(best)
    slot = np.where(found, team_slots[idx] + 1, -1)
    return slot, np.where(found, best, np.nan)


def build(atd_path, meta_path, frames_out):
    """The clean table for frames_out, plus a dict of aggregate diagnostics."""
    teams = team_of_each_track(meta_path)
    n_slots = len(teams)
    home_slots = np.flatnonzero(teams == HOME)
    away_slots = np.flatnonzero(teams != HOME)

    # Read a margin around the output frames: one frame before for ball speed, and FROZEN_FRAMES-1
    # each side so a freeze crossing the edge of the range is measured at its full length.
    n_video = games.source_frames("tactical")
    margin = C.FROZEN_FRAMES - 1
    wanted = np.zeros(n_video, dtype=bool)
    for lo, hi in _blocks(frames_out):
        wanted[max(0, lo - margin):min(n_video, hi + margin + 1)] = True

    vf, players, ball, malformed = read_tracking(atd_path, wanted, n_slots)

    frozen = frozen_mask(vf, players, C.FROZEN_FRAMES)
    present = ~np.isnan(players[:, :, 0])
    usable = present & ~frozen

    # Ball speed against the previous video frame.
    prev_ok = np.concatenate(([False], np.diff(vf) == 1))
    step = np.full(len(vf), np.nan)
    step[1:] = np.hypot(ball[1:, 0] - ball[:-1, 0], ball[1:, 1] - ball[:-1, 1])
    speed = np.where(prev_ok, step * C.FPS, np.nan)

    h_slot, h_dist = nearest(players, ball, usable, home_slots)
    a_slot, a_dist = nearest(players, ball, usable, away_slots)
    h_slot_all, _ = nearest(players, ball, present, home_slots)
    a_slot_all, _ = nearest(players, ball, present, away_slots)

    # Keep only output frames; frames with no tracking line at all stay as empty rows.
    pos = {v: i for i, v in enumerate(vf.tolist())}
    rows = np.array([pos.get(v, -1) for v in frames_out.tolist()], dtype=np.int64)
    have = rows >= 0
    r = np.where(have, rows, 0)

    def pick(arr, fill):
        return np.where(have, arr[r], fill)

    vfs = frames_out.tolist()
    half = np.array([T.half(v) for v in vfs])
    table = pd.DataFrame({
        "video_frame": frames_out,
        "half": half,
        "video_time": [T.video_time(v) for v in vfs],
        "match_time": [T.match_time(v) for v in vfs],
        "ball_x_m": pick(ball[:, 0], np.nan),
        "ball_y_m": pick(ball[:, 1], np.nan),
        "ball_speed": pick(speed, np.nan),
        "near_h_slot": pd.array(np.where(pick(h_slot, -1) > 0, pick(h_slot, -1), 0), dtype="Int64"),
        "near_h_dist": pick(h_dist, np.nan),
        "near_a_slot": pd.array(np.where(pick(a_slot, -1) > 0, pick(a_slot, -1), 0), dtype="Int64"),
        "near_a_dist": pick(a_dist, np.nan),
    }, columns=COLUMNS)
    table.loc[table["near_h_slot"] == 0, "near_h_slot"] = pd.NA
    table.loc[table["near_a_slot"] == 0, "near_a_slot"] = pd.NA

    frozen_out = frozen[r] & have[:, None]
    ball_seen = pick(~np.isnan(ball[:, 0]), False)
    changed = ball_seen & ((pick(h_slot, -1) != pick(h_slot_all, -1))
                           | (pick(a_slot, -1) != pick(a_slot_all, -1)))
    stats = {
        "malformed_lines": malformed,
        "frames_without_line": int((~have).sum()),
        "frozen_slot_frames": int(frozen_out.sum()),
        "frames_with_frozen_slot": int(frozen_out.any(axis=1).sum()),
        "frames_nearest_changed_by_frozen": int(changed.sum()),
        "frames_ball_but_no_h": int((ball_seen & (pick(h_slot, -1) < 0)).sum()),
        "frames_ball_but_no_a": int((ball_seen & (pick(a_slot, -1) < 0)).sum()),
    }
    return table, stats


def _blocks(frames_sorted):
    """Contiguous [lo, hi] runs of a sorted frame array."""
    if not len(frames_sorted):
        return []
    breaks = np.flatnonzero(np.diff(frames_sorted) != 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(frames_sorted) - 1]))
    return [(int(frames_sorted[s]), int(frames_sorted[e])) for s, e in zip(starts, ends)]


def report(table, stats, elapsed, out_path):
    """Aggregates only (counts, shares, percentiles); no per-frame values."""
    blocks = {h: (first, last) for h, first, last in T.graded_blocks()}
    print("wrote %d rows to %s" % (len(table), out_path))
    for h, part in table.groupby("half"):
        first, last = blocks[h]
        expected = last - first + 1
        verdict = "OK" if len(part) == expected else "partial selection"
        print("  half %d: %6d rows (full half = %d: %s)   ball missing %.1f%%" % (
            h, len(part), expected, verdict, 100.0 * part["ball_x_m"].isna().mean()))
    ends = table.iloc[[0, -1]][["video_frame", "half", "video_time", "match_time"]]
    print("  first row: video %d  half %d  %s  match %s" % tuple(ends.iloc[0]))
    print("  last row:  video %d  half %d  %s  match %s" % tuple(ends.iloc[1]))

    sp = table["ball_speed"].dropna().to_numpy()
    if len(sp):
        p50, p90, p99 = np.percentile(sp, [50, 90, 99])
        print("  ball speed m/s over %d frames: p50 %.2f  p90 %.2f  p99 %.2f  max %.1f   "
              "> %.0f m/s: %d frames" % (len(sp), p50, p90, p99, sp.max(), C.SANE_BALL_SPEED,
                                         int((sp > C.SANE_BALL_SPEED).sum())))
    print("  frozen slots (identical over %d frames, or a ghost: still stretch, path < %.2f m, with a "
          "live box within %.0f m when it started): "
          "%d slot-frames excluded, in %d frames; nearest H/A slot changed by the exclusion in %d frames" % (
              C.FROZEN_FRAMES, C.FROZEN_MAX_PATH_M, C.FROZEN_TWIN_RADIUS_M, stats["frozen_slot_frames"],
              stats["frames_with_frozen_slot"],
              stats["frames_nearest_changed_by_frozen"]))
    print("  ball seen but no usable slot: H %d frames, A %d frames" % (
        stats["frames_ball_but_no_h"], stats["frames_ball_but_no_a"]))
    print("  malformed lines skipped: %d   output frames with no tracking line: %d" % (
        stats["malformed_lines"], stats["frames_without_line"]))
    print("  run time %.1f s" % elapsed)


def main():
    paths = games.asset_paths(source="tactical")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atd", default=paths["atd_raw"], help="raw tracking .txt")
    ap.add_argument("--meta", default=paths["atd_meta"], help="tracking metadata .xml")
    ap.add_argument("--half", type=int, choices=(1, 2), help="only this half")
    ap.add_argument("--start", type=int, help="first video frame (inclusive)")
    ap.add_argument("--end", type=int, help="last video frame (inclusive)")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "clean_table.csv"))
    a = ap.parse_args()

    for name in ("atd", "meta"):
        if not os.path.isfile(getattr(a, name)):
            fail_missing(name, getattr(a, name))

    t0 = time.time()
    frames_out = selected_frames(a.half, a.start, a.end)
    if not len(frames_out):
        sys.exit("No graded frames in that selection.")
    table, stats = build(a.atd, a.meta, frames_out)

    folder = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(folder, exist_ok=True)
    table.to_csv(a.out, index=False, float_format="%.3f", lineterminator="\n")
    report(table, stats, time.time() - t0, a.out)


if __name__ == "__main__":
    main()
