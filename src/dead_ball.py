"""Step 6: dead-ball stretches (D), as an override for the step 5 labels.

Missing ball alone never creates D.

1. Ball out: the ball is seen beyond a line by more than the margin -- x < -OUT_MARGIN (ball_out_x<0),
   x > PITCH_L + OUT_MARGIN (ball_out_x>L), y > PITCH_W + OUT_MARGIN (ball_out_y_near),
   y < -FAR_OUT_MARGIN (ball_out_y_far; the far touchline is less accurate). D from that frame until
   the restart touch: the first `own` onset (out/events.csv) after the ball is seen back inside; if a
   speed_up touch onset comes first (a restart kick with no attributed toucher), that one.
2. Before kick-off: from each half's first graded frame, D until the kick-off = the first speed_up
   touch within KICKOFF_SEARCH_FRAMES whose onset has the ball within KICKOFF_SPOT_RADIUS_M of the
   centre spot (before_kickoff).
   After a goal: a later speed_up touch from the centre spot with >= KICKOFF_OWN_HALF_SHARE of the
   usable players in their own half (a kick-off, not a pass through the spot) -> D from the last
   frame, within AFTER_GOAL_MAX_FRAMES before it, where the ball was seen in a goal area or out over a
   goal line, until that kick-off (after_goal).
3. Smart Tagging SET PIECES windows only confirm (overlap) or extend an existing D stretch: a
   confirmed stretch whose restart was not an attributed own is extended to the first own onset
   inside the window (set_piece_confirmed). A window never creates D alone. --no-tags skips this.
4. D stretches shorter than MIN_DEAD_FRAMES are dropped (a restart takes longer; short D would only
   split a real possession).

Output: out/labels_dead.csv (video_frame, label D or empty, reason). Stdout: aggregates only.

    python src/dead_ball.py
    python src/dead_ball.py --show <vf> ... --around 50   # per-frame rows for local debugging
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
    sys.exit("Can't find games.py. Run this from the repo root: python src/dead_ball.py")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import games                                            # noqa: E402
from starter.baseline import team_of_each_track, HOME   # noqa: E402
import timeutil as T                                    # noqa: E402
import config as C                                      # noqa: E402
from clean_table import read_tracking, frozen_mask      # noqa: E402
from team_check import home_defends_first_half, LEFT    # noqa: E402

# Priority when stretches overlap: the more specific explanation wins.
REASON_PRIORITY = ["before_kickoff", "after_goal", "ball_out_x<0", "ball_out_x>L",
                   "ball_out_y_near", "ball_out_y_far", "restart_throw_in", "restart_corner",
                   "set_piece_confirmed"]


def load(p):
    ct = pd.read_csv(p.table, usecols=["video_frame", "half", "ball_x_m", "ball_y_m", "ball_speed"])
    glitch = (ct["ball_speed"] > C.SANE_BALL_SPEED).to_numpy()
    D = {"vf": ct["video_frame"].to_numpy(), "half": ct["half"].to_numpy(),
         "bx": np.where(glitch, np.nan, ct["ball_x_m"].to_numpy()),
         "by": np.where(glitch, np.nan, ct["ball_y_m"].to_numpy()),
         "speed": np.where(glitch, np.nan, ct["ball_speed"].to_numpy())}
    D["row_of"] = {v: i for i, v in enumerate(D["vf"].tolist())}
    ev = pd.read_csv(p.events)
    own = ev[ev["event"] == "own"]
    D["own_rows"] = np.array(sorted({D["row_of"][v] for v in own["onset_frame"] if v in D["row_of"]}),
                             dtype=np.int64)
    tt = pd.read_csv(p.touches)
    up = tt[tt["kind"] == "speed_up"]
    D["kick_rows"] = np.array(sorted({D["row_of"][v] for v in up["onset_frame"] if v in D["row_of"]}),
                              dtype=np.int64)
    sg = tt[tt["kind"] == "stop_go"]
    D["stopgo_rows"] = np.array(sorted({D["row_of"][v] for v in sg["onset_frame"] if v in D["row_of"]}),
                                dtype=np.int64)
    D["blocks"] = []
    for h, first, last in T.graded_blocks():
        rows = np.flatnonzero(D["half"] == h)
        if len(rows):
            D["blocks"].append((h, int(rows[0]), int(rows[-1]) + 1))
    return D


def out_reasons(D):
    """Per-row reason string if the ball is seen out, else ''."""
    x, y = D["bx"], D["by"]
    r = np.array([""] * len(x), dtype=object)
    with np.errstate(invalid="ignore"):
        r[y < -C.FAR_OUT_MARGIN] = "ball_out_y_far"
        r[y > C.PITCH_W + C.OUT_MARGIN] = "ball_out_y_near"
        r[x > C.PITCH_L + C.OUT_MARGIN] = "ball_out_x>L"
        r[x < -C.OUT_MARGIN] = "ball_out_x<0"
    return r


def next_after(sorted_rows, start, stop):
    """First value in sorted_rows within [start, stop), or None."""
    k = np.searchsorted(sorted_rows, start)
    return int(sorted_rows[k]) if k < len(sorted_rows) and sorted_rows[k] < stop else None


def tag_overlaps(windows, s, e, types=None):
    """True if a SET PIECES window (of one of `types`, lower case; any Type if None) overlaps [s, e)."""
    return any(ws < e and s < we and (types is None or typ.lower() in types) for ws, we, typ in windows)


def ball_out_stretches(D, stats, windows):
    """[(start, end_exclusive, reason, restart_kind)] for the ball-out rule.

    A far-side out sighting is confirmed if the ball stays beyond the far line on >=
    FAR_OUT_SUSTAIN_FRAMES frames, OR a SET PIECES window (any Type) overlaps the stretch: the ball was
    seen out either way, the tag only confirms. THROWIN_MIN_FRAMES applies only to touchline stretches
    with no tag confirmation; a confirmed stretch runs to its restart touch.
    """
    reasons = out_reasons(D)
    seen = ~np.isnan(D["bx"])
    inside = seen & (reasons == "")
    out = []
    for h, a, b in D["blocks"]:
        i = a
        while i < b:
            if not reasons[i]:
                i += 1
                continue
            back = i + 1 + int(np.argmax(inside[i + 1:b])) if inside[i + 1:b].any() else b
            r = reasons[i]
            own = next_after(D["own_rows"], back, b)
            kick = next_after(D["kick_rows"], back, b)
            if own is not None and (kick is None or own <= kick):
                end, how = own, "own"
            elif kick is not None:
                end, how = kick, "kick"
            else:
                end, how = b, "half_end"
            end = max(end, i + 1)
            touchline = r in ("ball_out_y_near", "ball_out_y_far")
            tagged = touchline and tag_overlaps(windows, i, end)
            if r == "ball_out_y_far":
                sustained = int((reasons[i:back] == "ball_out_y_far").sum()) >= C.FAR_OUT_SUSTAIN_FRAMES
                if not sustained and not tagged:
                    stats["far_not_confirmed"] = stats.get("far_not_confirmed", 0) + 1
                    i = back
                    continue
                if not sustained:
                    stats["far_confirmed_by_tag"] = stats.get("far_confirmed_by_tag", 0) + 1
            # an unconfirmed touchline stoppage shorter than THROWIN_MIN_FRAMES can't be a throw-in
            if touchline and not tagged and end - i < C.THROWIN_MIN_FRAMES:
                key = "throwin_short_" + ("near" if r == "ball_out_y_near" else "far")
                stats[key] = stats.get(key, 0) + 1
                i = max(end, back, i + 1)
                continue
            stats["restart_" + how] = stats.get("restart_" + how, 0) + 1
            out.append((i, end, reasons[i], how))
            stats.setdefault("out_detail", []).append((h, i, back, end, reasons[i], how))
            i = max(end, back)
    return out


def restart_spot_stretches(D, windows, stats):
    """Throw-ins and corners the tracker didn't see go out (relaxed restart-spot rule).

    After a missing-ball gap >= RESTART_GAP_FRAMES, the ball reappears (any speed) within
    RESTART_SPOT_M of a touchline (throw-in) or within RESTART_CORNER_M of a corner (corner), and a
    SET PIECES window of that Type overlaps the gap or the reappearance. Then D runs from
    max(last sighting before the gap, reappearance - RESTART_MAX_DEAD_FRAMES) to the reappearance
    kick: the onset of a speed_up / stop_go touch within MAX_GAP_FRAMES (1 s) after the
    reappearance if there is one, else the reappearance itself (the ball usually reappears already
    played). Goal kicks are not applied (too weakly confirmed by tags: 3 of 12 in the measurement).
    """
    seen = ~np.isnan(D["bx"])
    corners = [(0.0, 0.0), (0.0, C.PITCH_W), (C.PITCH_L, 0.0), (C.PITCH_L, C.PITCH_W)]
    restart_rows = np.array(sorted(set(D["kick_rows"].tolist()) | set(D["stopgo_rows"].tolist())),
                            dtype=np.int64)
    out = []
    for h, a, b in D["blocks"]:
        m = (~seen[a:b]).astype(np.int8)
        e = np.diff(np.concatenate(([0], m, [0])))
        for s0, t0 in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            g, r = a + s0 - 1, a + t0
            if t0 - s0 < C.RESTART_GAP_FRAMES or g < a or r >= b:
                continue
            x, y = D["bx"][r], D["by"][r]
            if min(np.hypot(x - qx, y - qy) for qx, qy in corners) <= C.RESTART_CORNER_M:
                kind, typ = "restart_corner", {"corner kick"}
            elif y <= C.RESTART_SPOT_M or y >= C.PITCH_W - C.RESTART_SPOT_M:
                kind, typ = "restart_throw_in", {"throw-in"}
            else:
                continue
            if not tag_overlaps(windows, g + 1, r + 1, typ):
                stats["restart_untagged_" + kind] = stats.get("restart_untagged_" + kind, 0) + 1
                continue
            k = next_after(restart_rows, r, min(b, r + C.MAX_GAP_FRAMES + 1))
            end = k if k is not None else r
            start = max(g, r - C.RESTART_MAX_DEAD_FRAMES)
            if end > start:
                out.append((start, end, kind, "restart_spot"))
    return out


def lined_up(D, track):
    """[n] True where >= KICKOFF_OWN_HALF_SHARE of the usable players stand in their own half: the
    kick-off line-up. A frame with fewer than KICKOFF_MIN_PLAYERS usable players cannot be judged and
    continues the previous frame's state, so a few dropped boxes don't cut a line-up in two."""
    players, usable, team_code, home_goal_h1 = track
    mid = C.PITCH_L / 2
    x = players[:, :, 0]
    u = usable & ~np.isnan(x)
    home_left = ((home_goal_h1 == LEFT) == (D["half"] == 1))[:, None]
    home = team_code == 0
    own = np.where(home == home_left, x <= mid + C.HALFWAY_TOL_M, x >= mid - C.HALFWAY_TOL_M)
    count = u.sum(axis=1)
    share = (own & u).sum(axis=1) / np.maximum(count, 1)
    known = count >= C.KICKOFF_MIN_PLAYERS
    state = (share >= C.KICKOFF_OWN_HALF_SHARE) & known
    idx = np.where(known, np.arange(len(state)), -1)
    last = np.maximum.accumulate(idx)                        # last judged frame at or before each row
    return np.where(last >= 0, state[np.maximum(last, 0)], False)


def kickoff_stretches(D, track, stats):
    """before_kickoff and after_goal stretches.

    Kick-off = the end of a line-up (lined_up() for >= KICKOFF_LINEUP_FRAMES): the ball is played
    and the players leave their halves. This needs no ball tracking -- the ball is often not seen
    until it is already moving. Half start: the first line-up beginning within KICKOFF_SEARCH_FRAMES.
    Dips shorter than KICKOFF_LINEUP_FRAMES inside a line-up (players stepping onto the halfway line,
    a box flickering) are bridged.
    After a goal: a later line-up with the ball seen within KICKOFF_SPOT_RADIUS_M of the centre spot;
    D from the last six-yard-box or goal-line sighting before the line-up; if none, the last
    penalty-area sighting; both searched no earlier than AFTER_GOAL_MAX_FRAMES (90 s: a goal-to-restart
    takes ~40-90 s) before the line-up; if neither, from the line-up start; until the kick-off.
    """
    cx, cy = C.PITCH_L / 2, C.PITCH_W / 2
    with np.errstate(invalid="ignore"):
        at_spot = np.hypot(D["bx"] - cx, D["by"] - cy) <= C.KICKOFF_SPOT_RADIUS_M
        in_pen = (np.abs(D["by"] - cy) <= C.PENALTY_AREA_WIDTH_M / 2) & \
            ((D["bx"] <= C.PENALTY_AREA_DEPTH_M) | (D["bx"] >= C.PITCH_L - C.PENALTY_AREA_DEPTH_M))
        in_six = (np.abs(D["by"] - cy) <= C.GOAL_AREA_WIDTH_M / 2) & \
            ((D["bx"] <= C.GOAL_AREA_DEPTH_M) | (D["bx"] >= C.PITCH_L - C.GOAL_AREA_DEPTH_M))
        over_line = (D["bx"] < -C.OUT_MARGIN) | (D["bx"] > C.PITCH_L + C.OUT_MARGIN)
    six_zone = in_six | over_line
    lu = bridge_dips(lined_up(D, track), D["blocks"], C.KICKOFF_LINEUP_FRAMES).astype(np.int8)
    out = []
    for h, a, b in D["blocks"]:
        e = np.diff(np.concatenate(([0], lu[a:b], [0])))
        lineups = [(a + s, a + t) for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))
                   if t - s >= C.KICKOFF_LINEUP_FRAMES]
        first = next(((s, t) for s, t in lineups if s < a + C.KICKOFF_SEARCH_FRAMES), None)
        stats["kickoff"][h] = None if first is None else (first[1] - a) / C.FPS
        if first is not None and first[1] > a:
            out.append((a, first[1], "before_kickoff", "kickoff"))
        for s, t in lineups:
            if first is not None and s <= first[0]:
                continue
            if not at_spot[s:t].any():
                stats["lineup_without_ball_at_spot"] += 1
                continue
            lo = max(a, s - C.AFTER_GOAL_MAX_FRAMES)
            six_hits = np.flatnonzero(six_zone[lo:s])
            pen_hits = np.flatnonzero(in_pen[lo:s])
            if len(six_hits):
                start, src = lo + int(six_hits[-1]), "six-yard box / goal line"
            elif len(pen_hits):
                start, src = lo + int(pen_hits[-1]), "penalty area"
            else:
                start, src = s, "line-up start"
                stats["after_goal_no_goal_zone"] += 1
            out.append((start, t, "after_goal", "kickoff"))
            stats["after_goal_times"].append(T.match_time(int(D["vf"][t])))
            stats["after_goal_kickoff_rows"].append(t)
            stats["after_goal_len"].append(((t - start) / C.FPS, src))
    return out


def bridge_dips(mask, blocks, max_gap):
    """mask with False runs shorter than max_gap filled where they lie between True frames of the
    same half (players stepping onto the halfway line, a box flickering)."""
    out = mask.copy()
    for _, a, b in blocks:
        m = mask[a:b].astype(np.int8)
        e = np.diff(np.concatenate(([0], m, [0])))
        starts, ends = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
        for k in range(len(starts) - 1):
            if starts[k + 1] - ends[k] < max_gap:
                out[a + ends[k]:a + starts[k + 1]] = True
    return out


def tag_windows(p, D):
    """[(start_row, end_row_exclusive, type)] of SET PIECES windows inside the graded halves."""
    sys.path.insert(0, os.path.join(REPO, "starter"))
    from load_smart_tagging import load as load_tags
    out = []
    for e in load_tags():
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        rows = [D["row_of"][v] for v in range(e["start_frame"], e["end_frame"]) if v in D["row_of"]]
        if rows:
            out.append((min(rows), max(rows) + 1, (e["tags"].get("Type") or "?").strip()))
    return out


def paint(n, stretches):
    """label/reason arrays from stretches, highest-priority reason per frame."""
    lab = np.array([""] * n, dtype=object)
    reason = np.array([""] * n, dtype=object)
    rank = np.full(n, len(REASON_PRIORITY), dtype=np.int64)
    for s, e, r, _ in stretches:
        pr = REASON_PRIORITY.index(r)
        seg = slice(s, e)
        better = rank[seg] > pr
        lab[seg] = np.where(better, "D", lab[seg])
        reason[seg] = np.where(better, r, reason[seg])
        rank[seg] = np.minimum(rank[seg], pr)
    return lab, reason


def runs_of(lab, half=None):
    """[(start, stop)] runs of D; with `half`, a run never spans the boundary between halves (the
    rows of both halves are adjacent in the arrays)."""
    m = (lab == "D").astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    out = list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
    if half is None:
        return out
    split = []
    for s, e in out:
        cuts = [k for k in range(s + 1, e) if half[k] != half[k - 1]]
        for a, b in zip([s] + cuts, cuts + [e]):
            split.append((a, b))
    return split


def checks(D, lab, reason, stretches, stats, windows, p):
    """Report-only checks 5-8. Aggregates (and match times to inspect with --show) only."""
    n = len(lab)
    is_d = lab == "D"
    runs = runs_of(lab, D["half"])

    # 5. far-touchline stretches that survived into the output
    print("check 5 -- far-touchline (y < -%.1f m) D stretches:" % C.FAR_OUT_MARGIN)
    for h, a, b in D["blocks"]:
        rows = [d for d in stats.get("out_detail", [])
                if d[0] == h and d[4] == "ball_out_y_far" and is_d[d[1]]]
        if not rows:
            print("  half %d: none" % h)
            continue
        lens, depths, throw, other, unseen, flagged = [], [], 0, 0, 0, []
        for _, s, back, e, _, how in rows:
            lens.append((e - s) / C.FPS)
            seg = -D["by"][s:back]
            depths.append(float(np.nanmax(seg)) if np.isfinite(seg).any() else 0.0)
            r = e if e < n else n - 1
            if how == "half_end" or np.isnan(D["by"][r]):
                unseen += 1
            elif D["by"][r] <= C.THROWIN_NEAR_LINE_M:
                throw += 1
            else:
                other += 1
            if lens[-1] < C.FAR_OUT_CHECK_MIN_S or depths[-1] < C.FAR_OUT_CHECK_MIN_DEPTH_M:
                flagged.append(T.match_time(int(D["vf"][s])))
        L, Dp = np.array(lens), np.array(depths)
        print("  half %d: %d stretches; length s p10 %.1f p50 %.1f p90 %.1f max %.0f; beyond the line "
              "m p50 %.1f max %.1f; restart within %.0f m of the touchline (throw-in) %d, elsewhere %d, "
              "restart ball unseen or half end %d" % (
                  h, len(rows), *np.percentile(L, [10, 50, 90]), L.max(), np.median(Dp), Dp.max(),
                  C.THROWIN_NEAR_LINE_M, throw, other, unseen))
        print("    flagged (< %.0f s or < %.0f m beyond the line): %d%s" % (
            C.FAR_OUT_CHECK_MIN_S, C.FAR_OUT_CHECK_MIN_DEPTH_M, len(flagged),
            (" -- starting at match " + ", ".join(flagged)) if flagged else ""))

    # 6. Smart Tagging SET PIECES windows by Type
    if windows:
        print("check 6 -- SET PIECES windows matching a D stretch, by Type:")
        by_type = {}
        for ws, we, typ in windows:
            m, t = by_type.get(typ, (0, 0))
            by_type[typ] = (m + int(is_d[ws:we].any()), t + 1)
        for typ, (m, t) in sorted(by_type.items()):
            print("  %-24s %2d of %2d" % (typ, m, t))
        tol = int(C.TAG_MATCH_S * C.FPS)
        ag_tags = [(ws, we) for ws, we, typ in windows if "after goal" in typ.lower()]
        kos = stats["after_goal_kickoff_rows"]
        tag_hit = sum(1 for ws, we in ag_tags if any(ws - tol <= k <= we + tol for k in kos))
        ko_hit = sum(1 for k in kos if any(ws - tol <= k <= we + tol for ws, we in ag_tags))
        print("  'Kick Off (After Goal)' tags: %d; %d lie within %.0f s of a detected after-goal "
              "kick-off; detected after-goal kick-offs with a tag: %d of %d" % (
                  len(ag_tags), tag_hit, C.TAG_MATCH_S, ko_hit, len(kos)))

    # 7. same owner before and after a D stretch
    lr = pd.read_csv(p.labels, keep_default_na=False).set_index("video_frame").reindex(D["vf"])
    raw = lr["label"].to_numpy()
    same, diff, unknown, by = 0, 0, 0, {}
    for s, e in runs:
        before = raw[s - 1] if s > 0 and D["half"][s - 1] == D["half"][s] else ""
        after = raw[e] if e < n and D["half"][e] == D["half"][s] else ""
        key = reason[s]
        c = by.setdefault(key, [0, 0, 0])
        if not before or not after:
            unknown += 1
            c[2] += 1
        elif before == after:
            same += 1
            c[0] += 1
        else:
            diff += 1
            c[1] += 1
    print("check 7 -- owner before vs after each D stretch (step 5 labels): same %d, different %d, "
          "unknown %d (%.0f%% same of known)" % (same, diff, unknown,
                                                 100.0 * same / max(1, same + diff)))
    print("  by reason (same / different / unknown): %s" % ", ".join(
        "%s %d/%d/%d" % (k, *v) for k, v in sorted(by.items())))

    # 8. every D frame comes from a ball-out or line-up trigger
    reasons_out = out_reasons(D)
    ball_out_ends = {e for s, e, r, how in stretches if r.startswith("ball_out")}
    kinds = {"ball-out sighting": np.zeros(n, dtype=bool), "line-up": np.zeros(n, dtype=bool),
             "restart spot + tag": np.zeros(n, dtype=bool)}
    for s, e, r, how in stretches:
        if r.startswith("ball_out"):
            if bool(reasons_out[s]) and not np.isnan(D["bx"][s]):          # ball seen out at the start
                kinds["ball-out sighting"][s:e] = True
        elif r in ("before_kickoff", "after_goal"):
            if how == "kickoff":                                              # ends at a line-up end
                kinds["line-up"][s:e] = True
        elif r.startswith("restart_"):
            if how == "restart_spot":                                         # spot + same-Type tag
                kinds["restart spot + tag"][s:e] = True
        elif s in ball_out_ends:                                              # set_piece_confirmed
            kinds["ball-out sighting"][s:e] = True
    ok = np.zeros(n, dtype=bool)
    for m in kinds.values():
        ok |= m
    print("check 8 -- D frames with positive evidence: %.1f%% (%d of %d); by evidence (a frame can have "
          "several): %s" % (100.0 * (ok & is_d).sum() / max(1, is_d.sum()), int((ok & is_d).sum()),
                            int(is_d.sum()), ", ".join("%s %d" % (k, int((m & is_d).sum()))
                                                       for k, m in kinds.items())))


def restart_spot_report(spot, stats):
    """Aggregates for the restart-spot rule (stretches as built, before overlap painting)."""
    print("restart-spot rule (ball gap >= %d s, reappearing at a touchline / corner, confirmed by a "
          "tag of that Type):" % (C.RESTART_GAP_FRAMES // C.FPS))
    for kind in ("restart_throw_in", "restart_corner"):
        L = np.array([(e - s) / C.FPS for s, e, r, _ in spot if r == kind])
        if len(L):
            print("  %-17s %d stretches; length s p10 %.1f p50 %.1f p90 %.1f max %.0f; at the spot but "
                  "no tag of that Type %d" % (kind, len(L), *np.percentile(L, [10, 50, 90]), L.max(),
                                              stats.get("restart_untagged_" + kind, 0)))
        else:
            print("  %-17s 0 (at the spot but no tag of that Type %d)" % (
                kind, stats.get("restart_untagged_" + kind, 0)))


def main():
    out = os.path.join(REPO, "out")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=os.path.join(out, "clean_table.csv"))
    ap.add_argument("--events", default=os.path.join(out, "events.csv"))
    ap.add_argument("--touches", default=os.path.join(out, "touches.csv"))
    ap.add_argument("--keepers", default=os.path.join(out, "keepers.csv"))
    ap.add_argument("--labels", default=os.path.join(out, "labels_raw.csv"),
                    help="step 5 labels, for the owner-before/after check only")
    ap.add_argument("--out", default=os.path.join(out, "labels_dead.csv"))
    ap.add_argument("--use-tags", dest="tags", action="store_true", default=True)
    ap.add_argument("--no-tags", dest="tags", action="store_false")
    ap.add_argument("--show", type=int, nargs="+", metavar="VF",
                    help="print per-frame rows around these video frames (match data)")
    ap.add_argument("--around", type=int, default=50)
    p = ap.parse_args()
    for name, hint in (("table", "step 1"), ("events", "step 5"), ("touches", "step 4"),
                       ("keepers", "step 3")):
        if not os.path.isfile(getattr(p, name)):
            sys.exit("Can't find the %s file (%s). Run %s first." % (name, getattr(p, name), hint))

    t0 = time.time()
    D = load(p)
    n = len(D["vf"])

    # players, usability and team (keeper override) for the kick-off own-half check
    paths = games.asset_paths(source="tactical")
    nominal = np.where(team_of_each_track(paths["atd_meta"]) == HOME, 0, 1).astype(np.int8)
    n_video = games.source_frames("tactical")
    wanted = np.zeros(n_video, dtype=bool)
    m = C.FROZEN_FRAMES - 1
    for _, first, last in T.graded_blocks():
        wanted[max(0, first - m):min(n_video, last + m + 1)] = True
    tvf, players, _, _ = read_tracking(paths["atd_raw"], wanted, len(nominal))
    usable = ~np.isnan(players[:, :, 0]) & ~frozen_mask(tvf, players, C.FROZEN_FRAMES)
    pos = np.searchsorted(tvf, D["vf"])
    players, usable = players[pos], usable[pos]
    team_code = np.tile(nominal, (n, 1))
    kp = pd.read_csv(p.keepers).set_index("video_frame").reindex(D["vf"])
    for code, col in ((0, "h_keeper_slot"), (1, "a_keeper_slot")):
        k = kp[col].to_numpy()
        r = np.flatnonzero(~np.isnan(k))
        team_code[r, k[r].astype(int) - 1] = code
    track = (players, usable, team_code, home_defends_first_half(paths["atd_meta"]))

    stats = {"kickoff": {}, "after_goal_no_goal_zone": 0, "lineup_without_ball_at_spot": 0,
             "after_goal_times": [], "after_goal_kickoff_rows": [], "after_goal_len": []}
    windows = tag_windows(p, D) if p.tags else []
    stretches = kickoff_stretches(D, track, stats) + ball_out_stretches(D, stats, windows)
    spot = restart_spot_stretches(D, windows, stats)
    stretches += spot

    confirmed, extended = 0, 0
    if p.tags:
        for ws, we, _ in windows:
            hit = [k for k, (s, e, r, how) in enumerate(stretches) if s < we and ws < e]
            if not hit:
                continue
            confirmed += 1
            for k in hit:
                s, e, r, how = stretches[k]
                if how != "own" and r.startswith("ball_out"):
                    own = next_after(D["own_rows"], max(e, ws), we)
                    if own is not None and own > e:
                        stretches.append((e, own, "set_piece_confirmed", "own"))
                        extended += 1

    lab, reason = paint(n, stretches)
    dropped = 0
    for s, e in runs_of(lab, D["half"]):
        if e - s < C.MIN_DEAD_FRAMES:
            lab[s:e], reason[s:e] = "", ""
            dropped += 1

    os.makedirs(os.path.dirname(os.path.abspath(p.out)), exist_ok=True)
    pd.DataFrame({"video_frame": D["vf"], "label": lab, "reason": reason}).to_csv(
        p.out, index=False, lineterminator="\n")

    # -- aggregates only ------------------------------------------------------------------------
    print("wrote %d rows to %s" % (n, p.out))
    lengths = []
    for h, a, b in D["blocks"]:
        rs = [(s, e) for s, e in runs_of(lab, D["half"]) if a <= s < b]
        by_reason = {}
        for s, e in rs:
            by_reason[reason[s]] = by_reason.get(reason[s], 0) + 1
            lengths.append((e - s) / C.FPS)
        ko = stats["kickoff"].get(h)
        print("half %d: D on %.1f%% of graded frames; %d D stretches (%s); kick-off %s" % (
            h, 100 * (lab[a:b] == "D").mean(), len(rs),
            ", ".join("%s %d" % kv for kv in sorted(by_reason.items())),
            "line-up not found in the first %d s (kick-off taken as the first graded frame; "
            "no before_kickoff D)" % C.KICKOFF_SEARCH_S
            if ko is None else "%.1f s after the half start" % ko))
    L = np.array(lengths)
    if len(L):
        print("D stretch length s: p10 %.1f  median %.1f  p90 %.1f  max %.0f" % (
            np.percentile(L, 10), np.median(L), np.percentile(L, 90), L.max()))
    print("after-goal kick-offs: %d (at match %s); %d later line-ups without the ball at the spot "
          "(ignored)" % (len(stats["after_goal_times"]), ", ".join(stats["after_goal_times"]) or "-",
                         stats["lineup_without_ball_at_spot"]))
    for tm, (length, src) in zip(stats["after_goal_times"], stats["after_goal_len"]):
        print("  after-goal D ending at match %s: %.1f s, starting at the last %s sighting" % (
            tm, length, src) if src != "line-up start" else
            "  after-goal D ending at match %s: %.1f s, starting at the line-up" % (tm, length))
    print("ball-out restarts: own %d, unattributed kick %d, none (to half end) %d" % (
        stats.get("restart_own", 0), stats.get("restart_kick", 0), stats.get("restart_half_end", 0)))
    print("far-side outs: confirmed by a tag although not sustained %d s %d; dropped (neither) %d; "
          "untagged touchline stoppages shorter than %d s dropped: near %d, far %d" % (
              C.FAR_OUT_SUSTAIN_TIME, stats.get("far_confirmed_by_tag", 0),
              stats.get("far_not_confirmed", 0), C.THROWIN_MIN_S,
              stats.get("throwin_short_near", 0), stats.get("throwin_short_far", 0)))
    print("D stretches dropped as shorter than %d frames: %d" % (C.MIN_DEAD_FRAMES, dropped))
    if p.tags:
        print("SET PIECES windows: %d; with a matching D stretch %d, without %d; stretches extended %d" % (
            len(windows), confirmed, len(windows) - confirmed, extended))
    print("D share of graded frames: %.1f%%" % (100 * (lab == "D").mean()))
    checks(D, lab, reason, stretches, stats, windows, p)
    restart_spot_report(spot, stats)
    print("run time %.1f s" % (time.time() - t0))

    if p.show:
        print("\n--show: per-frame rows for local debugging")
        reasons_out = out_reasons(D)
        own_set = set(D["own_rows"].tolist())
        kick_set = set(D["kick_rows"].tolist())
        win = np.zeros(n, dtype=bool)
        for ws, we, _ in windows:
            win[ws:we] = True
        for v in p.show:
            if v not in D["row_of"]:
                print("\nvideo %d: not in a graded half" % v)
                continue
            print("\nvideo %d  (half %d, video %s, match %s)" % (v, T.half(v), T.video_time(v),
                                                                 T.match_time(v)))
            print("   video  ball x,y (m)     out              tag  touch       label  reason")
            for x in range(v - p.around, v + p.around + 1):
                i = D["row_of"].get(x)
                if i is None:
                    continue
                ball = "   not seen   " if np.isnan(D["bx"][i]) else "%6.1f,%6.1f " % (D["bx"][i], D["by"][i])
                tch = "own" if i in own_set else ("kick" if i in kick_set else "")
                print("  %6d  %s  %-16s %-4s %-10s  %-5s  %s%s" % (
                    x, ball, reasons_out[i], "SP" if win[i] else "", tch, lab[i] or "-", reason[i],
                    "  <" if x == v else ""))


if __name__ == "__main__":
    main()
