"""Step 3: facts about the tracking slots, for step 5. Never changes a label.

1. Keeper detection, per graded frame and per goal:
   - Which goal each team defends comes from the metadata (`attack_direction_first_half`); the
     teams swap ends at halftime.
   - Candidate = the usable slot (present, not frozen; any nominal team) nearest that goal's
     centre, if within KEEPER_ZONE_M.
   - A candidate is the goal's keeper only if it is the candidate on at least KEEPER_PERSIST of a
     centred window of KEEPER_WINDOW_FRAMES (clipped to the half; at the clipped edges the same
     share of the frames that exist is required).
   - The keeper's team is the team defending that goal, whatever the slot's nominal team.
   -> out/keepers.csv: video_frame, h_keeper_slot, a_keeper_slot (empty = no keeper found)

2. Per slot and half: frames present, mean x / y (m), frames as H / A keeper, frozen stretches
   (count, frames; FROZEN_FRAMES rule), verdict.
   -> out/team_check.txt

3. The frames of out/clean_table.csv where the ball is seen but no usable home slot exists:
   stretches, longest stretch, how many fall in the first 30 s of a half.

Usage (from the repo root):
    python src/team_check.py                        # writes both files, prints aggregates only
    python src/team_check.py --show <vf> ...       # also prints per-frame rows for local debugging
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = next((d for d in (HERE, os.path.dirname(HERE), os.getcwd())
             if os.path.isfile(os.path.join(d, "games.py"))), None)
if REPO is None:
    sys.exit("Can't find games.py. Run this from the repo root: python src/team_check.py")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import games                                            # noqa: E402
from starter.baseline import team_of_each_track, HOME   # noqa: E402
import timeutil as T                                    # noqa: E402
import config as C                                      # noqa: E402
from clean_table import read_tracking, frozen_mask      # noqa: E402

LEFT, RIGHT = 0, 1                       # goal at x = 0 / x = PITCH_L
FIRST_SECONDS = 30                       # point 3: "within the first 30 s of a half"


def fail_missing(name, path):
    sys.exit("Can't find the %s file (%s). Put this script next to games.py, or pass "
             "--%s /path/to/file." % (name, path, name))


def home_defends_first_half(meta_path):
    """LEFT or RIGHT: the goal the home team (first <Team>) defends in the 1st half."""
    team = next(ET.parse(meta_path).getroot().iter("Team"))
    for p in team.iter("ProviderParameter"):
        if (p.findtext("Name") or "").strip().lower() == "attack_direction_first_half":
            v = (p.findtext("Value") or "").strip().lower().replace("-", "_").replace(" ", "_")
            if v.startswith("left"):
                return LEFT          # attacks left -> right, so defends the left goal
            if v.startswith("right"):
                return RIGHT
            sys.exit("Unrecognised attack_direction_first_half value in %s" % meta_path)
    sys.exit("No attack_direction_first_half for the home team in %s" % meta_path)


def defended_goal(home_goal_h1, half, team):
    """Goal (LEFT/RIGHT) defended by team 'H' or 'A' in that half."""
    g = home_goal_h1 if team == "H" else 1 - home_goal_h1
    return g if half == 1 else 1 - g


def runs(mask):
    """[(start, stop)] half-open runs of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def goal_candidates(players, usable, goal):
    """(slot index or -1, distance m) of the usable slot nearest the goal centre, within the zone."""
    gx = 0.0 if goal == LEFT else C.PITCH_L
    d = np.hypot(players[:, :, 0] - gx, players[:, :, 1] - C.PITCH_W / 2)
    d = np.where(usable & ~np.isnan(d), d, np.inf)
    idx = np.argmin(d, axis=1)
    best = d[np.arange(len(d)), idx]
    ok = best <= C.KEEPER_ZONE_M
    return np.where(ok, idx, -1), np.where(ok, best, np.nan)


def persistent(cand, n_slots):
    """(keeper slot index or -1, its candidate count in the window, frames needed) per frame."""
    n = len(cand)
    half_w = C.KEEPER_WINDOW_FRAMES // 2
    i = np.arange(n)
    lo = np.maximum(0, i - half_w)
    hi = np.minimum(n, i - half_w + C.KEEPER_WINDOW_FRAMES)       # window of WINDOW frames
    length = hi - lo
    # same share as PERSIST_FRAMES / WINDOW_FRAMES, in integers (ceil) so it is exact
    need = (C.KEEPER_PERSIST_FRAMES * length + C.KEEPER_WINDOW_FRAMES - 1) // C.KEEPER_WINDOW_FRAMES
    keeper = np.full(n, -1, dtype=np.int64)
    count = np.zeros(n, dtype=np.int64)
    for s in np.unique(cand[cand >= 0]):
        cs = np.concatenate(([0], np.cumsum(cand == s)))
        c = cs[hi] - cs[lo]
        hit = c >= need                         # > 50%, so at most one slot can hit
        keeper[hit] = s
        count[hit] = c[hit]
    return keeper, count, need


def assign_keepers(persist, players, usable, goal, other_persist):
    """Final keeper per frame with persistence, hysteresis and linking.

    persist   -- keeper from the persistence rule (slot index or -1): always wins
    keep      -- the established keeper stays while usable and within KEEPER_KEEP_ZONE_M of the goal
    handover  -- when the keeper's box disappears or freezes, the role links, for up to
                 KEEPER_LINK_MAX_FRAMES, to a usable box within reach = KEEPER_RUN_SPEED x elapsed
                 time + KEEPER_LINK_REACH_M of the keeper's last position and within
                 KEEPER_KEEP_ZONE_M of the goal: preferably one that appeared or unfroze inside the
                 window (the tracker re-boxing the keeper); otherwise the nearest such box, only if no
                 other usable box is within KEEPER_LINK_UNIQUE_M of it. Never the other team's
                 established (persistence) keeper. A linked box that leaves the keep zone within
                 KEEPER_LINK_UNDO_FRAMES was an outfield player: the link is undone (link_undone).
    none      -- no keeper
    (There is no stale-keeper backup rule: it would hand the role to attackers while a real
    keeper stands still on his line. Step 5's box-distribution rule covers the case.)
    Returns (keeper slot index or -1, source code, number of handovers, link stats).
    """
    PERSIST, KEEP, HANDOVER, NONE, LINK_UNDONE = 0, 1, 2, 3, 4
    n = len(persist)
    gx = 0.0 if goal == LEFT else C.PITCH_L
    gy = C.PITCH_W / 2
    keeper = np.full(n, -1, dtype=np.int64)
    source = np.full(n, NONE, dtype=np.int8)
    cur, last_pos, lost_at, lost_slot, handovers = -1, None, None, -1, 0
    link_start, link_slot = None, -1
    run_start = usable_run_start(usable)
    sstats = {"links": [], "undone": []}
    slot_goal = np.hypot(players[:, :, 0] - gx, players[:, :, 1] - gy)
    for i in range(n):
        p = persist[i]
        if p >= 0:
            cur, src = p, PERSIST
        elif cur >= 0 and usable[i, cur] and \
                np.hypot(players[i, cur, 0] - gx, players[i, cur, 1] - gy) <= C.KEEPER_KEEP_ZONE_M:
            src = KEEP
        else:
            if cur >= 0:
                # lost the role: a disappeared / frozen slot can be handed over, leaving the zone cannot
                if not usable[i, cur]:
                    lost_at, lost_slot = i, cur
                else:
                    lost_at = None
                    # (d) a freshly linked box that walks out of the keep zone was an outfield player
                    if cur == link_slot and link_start is not None and \
                            i - link_start <= C.KEEPER_LINK_UNDO_FRAMES:
                        keeper[link_start:i] = -1
                        source[link_start:i] = LINK_UNDONE
                        sstats["undone"].append((cur, link_start))
                        link_start = None
                cur = -1
            src = NONE
            if lost_at is not None and last_pos is not None and \
                    i - lost_at < C.KEEPER_LINK_MAX_FRAMES:
                ok = usable[i] & (slot_goal[i] <= C.KEEPER_KEEP_ZONE_M)   # (b) inside the keep zone
                ok[lost_slot] = False
                if other_persist[i] >= 0:
                    ok[other_persist[i]] = False            # never the other team's keeper
                reach = C.KEEPER_RUN_SPEED * (i - lost_at + 1) / C.FPS + C.KEEPER_LINK_REACH_M
                d = np.hypot(players[i, :, 0] - last_pos[0], players[i, :, 1] - last_pos[1])
                within = ok & ~np.isnan(d) & (d <= reach)
                fresh = within & (run_start[i] >= lost_at)  # (c) appeared / unfroze in the window
                j = -1
                if fresh.any():
                    j = int(np.argmin(np.where(fresh, d, np.inf)))
                elif within.any():
                    k = int(np.argmin(np.where(within, d, np.inf)))
                    dk = np.hypot(players[i, :, 0] - players[i, k, 0],
                                  players[i, :, 1] - players[i, k, 1])
                    crowd = usable[i] & ~np.isnan(dk) & (dk <= C.KEEPER_LINK_UNIQUE_M)
                    crowd[k] = False
                    if not crowd.any():                     # unambiguous: nobody else beside it
                        j = k
                if j >= 0:
                    cur, src, lost_at = j, HANDOVER, None
                    handovers += 1
                    link_start, link_slot = i, j
                    sstats["links"].append((j, i))
            elif lost_at is not None:
                lost_at = None
        if cur >= 0:
            keeper[i] = cur
            if not np.isnan(players[i, cur, 0]):      # a persistence keeper can be absent on a frame
                last_pos = players[i, cur]
            if src == PERSIST:
                lost_at = None
        if link_start is not None and (i - link_start > C.KEEPER_LINK_UNDO_FRAMES or
                                       (src == PERSIST and cur == link_slot) or cur != link_slot):
            link_start = None                         # the link stood (or was replaced)
        source[i] = src
    return keeper, source, handovers, sstats


def usable_run_start(usable):
    """[n, slots] first frame of the usable run each (frame, slot) belongs to; -1 if not usable.
    A run starting inside the link window means the box appeared or unfroze there."""
    n, slots = usable.shape
    idx = np.arange(n)[:, None]
    begins = usable & ~np.vstack([np.zeros((1, slots), bool), usable[:-1]])
    start = np.maximum.accumulate(np.where(begins, idx, -1), axis=0)
    return np.where(usable, start, -1)


SOURCE_NAMES = np.array(["persist", "keep", "handover", "none", "link_undone"])


def analyse(atd_path, meta_path):
    teams = team_of_each_track(meta_path)
    n_slots = len(teams)
    nominal = np.where(teams == HOME, "H", "A")
    home_goal_h1 = home_defends_first_half(meta_path)

    n_video = games.source_frames("tactical")
    margin = C.FROZEN_FRAMES - 1
    wanted = np.zeros(n_video, dtype=bool)
    for _, first, last in T.graded_blocks():
        wanted[max(0, first - margin):min(n_video, last + margin + 1)] = True
    vf, players, _, malformed = read_tracking(atd_path, wanted, n_slots)
    frozen, still, ghost = frozen_mask(vf, players, C.FROZEN_FRAMES, details=True)
    present = ~np.isnan(players[:, :, 0])
    usable = present & ~frozen

    halves = {}
    for h, first, last in T.graded_blocks():
        sel = np.flatnonzero((vf >= first) & (vf <= last))
        if len(sel) != last - first + 1:
            sys.exit("Tracking has %d lines for half %d, expected %d." % (len(sel), h, last - first + 1))
        res = {"vf": vf[sel], "present": present[sel], "frozen": frozen[sel],
               "still": still[sel], "ghost": ghost[sel], "usable": usable[sel],
               "players": players[sel]}
        for team in ("H", "A"):
            goal = defended_goal(home_goal_h1, h, team)
            cand, dist = goal_candidates(players[sel], usable[sel], goal)
            persist, count, need = persistent(cand, n_slots)
            res[team] = {"goal": goal, "cand": cand, "dist": dist, "persist": persist,
                         "count": count, "need": need}
        for team, other in (("H", "A"), ("A", "H")):
            keeper, source, handovers, sstats = assign_keepers(
                res[team]["persist"], players[sel], usable[sel], res[team]["goal"],
                res[other]["persist"])
            res[team].update({"keeper": keeper, "source": source, "handovers": handovers,
                              "links": sstats})
        halves[h] = res
    return halves, nominal, home_goal_h1, malformed


def slot_rows(halves, nominal):
    """Per half, per slot summary rows (dicts)."""
    out = {}
    for h, r in halves.items():
        rows = []
        for s in range(len(nominal)):
            pres = r["present"][:, s]
            n_pres = int(pres.sum())
            fz = runs(r["frozen"][:, s])
            fz_frames = int(sum(b - a for a, b in fz))
            hk = int((r["H"]["keeper"] == s).sum())
            ak = int((r["A"]["keeper"] == s).sum())
            own_k, other_k = (hk, ak) if nominal[s] == "H" else (ak, hk)
            if n_pres == 0:
                verdict = "absent"
            elif fz_frames >= C.MOSTLY_FROZEN_SHARE * n_pres:
                verdict = "mostly frozen"
            elif other_k > 0:
                verdict = "acted as other team's keeper"
            elif own_k > 0:
                verdict = "keeper"
            else:
                verdict = "ok"
            xy = r["players"][pres, s, :]
            rows.append({"slot": s + 1, "team": nominal[s], "present": n_pres,
                         "mean_x": float(xy[:, 0].mean()) if n_pres else np.nan,
                         "mean_y": float(xy[:, 1].mean()) if n_pres else np.nan,
                         "h_keeper": hk, "a_keeper": ak, "frozen_stretches": len(fz),
                         "frozen_frames": fz_frames, "verdict": verdict})
        out[h] = rows
    return out


def keeper_stats(halves, nominal, key="keeper"):
    """Aggregates for the report. key="persist" gives the persistence-only (before) numbers."""
    stats = {}
    for h, r in halves.items():
        n = len(r["vf"])
        s = {"frames": n}
        for team in ("H", "A"):
            k = r[team][key]
            have = k >= 0
            seq = k[have]
            gaps = runs(~have)
            s[team] = {
                "share": 100.0 * have.mean(),
                "slot_changes": int((seq[1:] != seq[:-1]).sum()) if len(seq) else 0,
                "gaps": len(gaps),
                "longest_gap": max((b - a for a, b in gaps), default=0),
                "distinct_slots": len(np.unique(seq)),
                "other_nominal_frames": int((nominal[seq] != team).sum()) if len(seq) else 0,
                "handovers": r[team]["handovers"] if key == "keeper" else 0,
                "sources": ({name: int((r[team]["source"] == i).sum())
                             for i, name in enumerate(SOURCE_NAMES)} if key == "keeper" else {}),
            }
        stats[h] = s
    return stats


def no_home_slot(table_path, halves):
    """Point 3, from clean_table.csv. Also a cross-check against the tracking just read."""
    t = pd.read_csv(table_path, usecols=["video_frame", "ball_x_m", "near_h_slot"])
    v = t.loc[t["ball_x_m"].notna() & t["near_h_slot"].isna(), "video_frame"].to_numpy()
    first_of = {h: first for h, first, _ in T.graded_blocks()}
    early = sum(1 for x in v.tolist() if x - first_of[T.half(x)] < C.frames(FIRST_SECONDS))
    if len(v):
        breaks = np.flatnonzero(np.diff(v) != 1)
        lengths = np.diff(np.concatenate(([0], breaks + 1, [len(v)])))
    else:
        lengths = np.array([0])
    # What the tracking shows in those frames: how many home slots are present / frozen.
    present_h, frozen_h = [], []
    for h, r in halves.items():
        idx = np.searchsorted(r["vf"], v[(v >= r["vf"][0]) & (v <= r["vf"][-1])])
        if len(idx):
            home = r["home_mask"]        # nominal home slots
            present_h.extend(r["present"][idx][:, home].sum(axis=1).tolist())
            frozen_h.extend((r["present"][idx] & r["frozen"][idx])[:, home].sum(axis=1).tolist())
    return {"frames": len(v), "stretches": int(len(lengths) if len(v) else 0),
            "longest": int(lengths.max()), "first_30s": early,
            "halves": {h: int(sum(1 for x in v.tolist() if T.half(x) == h)) for h in halves},
            "home_present_mean": float(np.mean(present_h)) if present_h else 0.0,
            "home_present_frozen_mean": float(np.mean(frozen_h)) if frozen_h else 0.0}


def write_keepers(halves, path):
    vfs, hk, ak, hs, as_ = [], [], [], [], []
    for h in sorted(halves):
        r = halves[h]
        vfs.append(r["vf"])
        hk.append(r["H"]["keeper"])
        ak.append(r["A"]["keeper"])
        hs.append(r["H"]["source"])
        as_.append(r["A"]["source"])
    hk, ak = np.concatenate(hk), np.concatenate(ak)
    table = pd.DataFrame({
        "video_frame": np.concatenate(vfs),
        "h_keeper_slot": pd.array(np.where(hk >= 0, hk + 1, 0), dtype="Int64"),
        "a_keeper_slot": pd.array(np.where(ak >= 0, ak + 1, 0), dtype="Int64"),
        "h_source": SOURCE_NAMES[np.concatenate(hs)],
        "a_source": SOURCE_NAMES[np.concatenate(as_)],
    })
    table.loc[table["h_keeper_slot"] == 0, "h_keeper_slot"] = pd.NA
    table.loc[table["a_keeper_slot"] == 0, "a_keeper_slot"] = pd.NA
    table.to_csv(path, index=False, lineterminator="\n")
    return len(table)


def write_report(path, rows, stats, home_goal_h1):
    side = {LEFT: "left goal (x = 0)", RIGHT: "right goal (x = %.1f m)" % C.PITCH_L}
    lines = ["Team check (step 3). Aggregates per slot and half; never changes a label.",
             "Home (H) defends the %s in half 1 and the %s in half 2." % (
                 side[home_goal_h1], side[1 - home_goal_h1]),
             "Keeper: nearest usable slot to the goal centre within %.1f m, on >= %d of %d frames "
             "of a centred window." % (C.KEEPER_ZONE_M, C.KEEPER_PERSIST_FRAMES,
                                       C.KEEPER_WINDOW_FRAMES),
             "Frozen stretch: total path < %.2f m over %d frames. 'mostly frozen': frozen >= %.0f%% "
             "of present frames." % (C.FROZEN_MAX_PATH_M, C.FROZEN_FRAMES,
                                     100 * C.MOSTLY_FROZEN_SHARE), ""]
    for h in sorted(rows):
        s = stats[h]
        lines.append("Half %d  (%d frames)   H keeper on %.1f%%, A keeper on %.1f%%" % (
            h, s["frames"], s["H"]["share"], s["A"]["share"]))
        lines.append("  slot team  present  mean_x  mean_y  h_keeper  a_keeper  frozen_n  "
                     "frozen_frames  verdict")
        for r in rows[h]:
            lines.append("  %4d  %s   %7d  %6s  %6s  %8d  %8d  %8d  %13d  %s" % (
                r["slot"], r["team"], r["present"],
                "" if np.isnan(r["mean_x"]) else "%.1f" % r["mean_x"],
                "" if np.isnan(r["mean_y"]) else "%.1f" % r["mean_y"],
                r["h_keeper"], r["a_keeper"], r["frozen_stretches"], r["frozen_frames"],
                r["verdict"]))
        lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def show(halves, frames_wanted, around):
    """Per-frame rows around the chosen frames, for local debugging; only when --show is given."""
    for v in frames_wanted:
        h = T.half(v)
        if h is None:
            print("\nvideo %d: not in a graded half" % v)
            continue
        r = halves[h]
        print("\nvideo %d  (half %d, video %s, match %s)" % (v, h, T.video_time(v), T.match_time(v)))
        print("  video   h_keeper  a_keeper   H-goal cand (dist m, count/need)   "
              "A-goal cand (dist m, count/need)")
        for x in range(v - around, v + around + 1):
            if T.half(x) != h:
                continue
            i = x - r["vf"][0]

            def fmt(team):
                c, d = r[team]["cand"][i], r[team]["dist"][i]
                cand = "-" if c < 0 else "slot %2d %5.1f" % (c + 1, d)
                return "%-16s %3d/%3d" % (cand, r[team]["count"][i], r[team]["need"][i])

            hk, ak = r["H"]["keeper"][i], r["A"]["keeper"][i]
            hs, as_ = SOURCE_NAMES[r["H"]["source"][i]], SOURCE_NAMES[r["A"]["source"][i]]
            print("  %6d  %8s  %8s   %-34s %s%s" % (
                x, ("-" if hk < 0 else "%d %s" % (hk + 1, hs[:4])),
                ("-" if ak < 0 else "%d %s" % (ak + 1, as_[:4])), fmt("H"), fmt("A"),
                "  <" if x == v else ""))
            # still / ghost boxes near each goal: slot, still-frames so far, nearest live box, frozen
            for team in ("H", "A"):
                gx = 0.0 if r[team]["goal"] == LEFT else C.PITCH_L
                P = r["players"][i]
                dg = np.hypot(P[:, 0] - gx, P[:, 1] - C.PITCH_W / 2)
                boxes = np.flatnonzero(r["still"][i] & (dg <= C.KEEPER_KEEP_ZONE_M))
                parts = []
                for s in boxes:
                    k = i
                    while k - 1 >= 0 and r["still"][k - 1, s]:
                        k -= 1
                    live = r["usable"][i] & ~r["still"][i]
                    live[s] = False
                    dt = np.hypot(P[:, 0] - P[s, 0], P[:, 1] - P[s, 1])
                    dt = np.where(live & ~np.isnan(dt), dt, np.inf)
                    parts.append("slot %d still %d f, twin %s, %s" % (
                        s + 1, i - k + 1, "%.1f m" % dt.min() if np.isfinite(dt.min()) else "-",
                        "GHOST/frozen" if r["frozen"][i, s] else "not frozen"))
                if parts:
                    print("          %s-goal still boxes: %s" % (team, "; ".join(parts)))


def main():
    paths = games.asset_paths(source="tactical")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atd", default=paths["atd_raw"], help="raw tracking .txt")
    ap.add_argument("--meta", default=paths["atd_meta"], help="tracking metadata .xml")
    ap.add_argument("--table", default=os.path.join(REPO, "out", "clean_table.csv"),
                    help="step 1 output, for point 3")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "team_check.txt"))
    ap.add_argument("--keepers-out", default=os.path.join(REPO, "out", "keepers.csv"))
    ap.add_argument("--show", type=int, nargs="+", metavar="VF",
                    help="print per-frame keeper rows around these video frames (match data)")
    ap.add_argument("--around", type=int, default=5, help="frames each side for --show")
    a = ap.parse_args()
    for name in ("atd", "meta", "table"):
        if not os.path.isfile(getattr(a, name)):
            if name == "table":
                sys.exit("Can't find the table file (%s). Run step 1 first: python src/clean_table.py"
                         % a.table)
            fail_missing(name, getattr(a, name))

    t0 = time.time()
    halves, nominal, home_goal_h1, malformed = analyse(a.atd, a.meta)
    for r in halves.values():
        r["home_mask"] = nominal == "H"
    rows = slot_rows(halves, nominal)
    stats = keeper_stats(halves, nominal)
    gap = no_home_slot(a.table, halves)

    for p in (a.out, a.keepers_out):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    n = write_keepers(halves, a.keepers_out)
    write_report(a.out, rows, stats, home_goal_h1)

    # -- aggregates only ----------------------------------------------------------------------
    print("wrote %d rows to %s and the slot summary to %s" % (n, a.keepers_out, a.out))
    print("home defends the %s goal in half 1" % ("left" if home_goal_h1 == LEFT else "right"))
    before = keeper_stats(halves, nominal, key="persist")
    print("keepers, persistence only (before) -> with keep + handover (after):")
    print("  half team  keeper%%        slot changes  handovers  other-nominal frames  "
          "longest gap (s)   sources after")
    for h in sorted(stats):
        for team in ("H", "A"):
            b, k = before[h][team], stats[h][team]
            print("  %d    %s     %5.1f -> %5.1f  %4d -> %4d   %5d      %5d -> %5d        "
                  "%5.1f -> %5.1f    %s" % (
                      h, team, b["share"], k["share"], b["slot_changes"], k["slot_changes"],
                      k["handovers"], b["other_nominal_frames"], k["other_nominal_frames"],
                      b["longest_gap"] / C.FPS, k["longest_gap"] / C.FPS,
                      " ".join("%s %d" % kv for kv in k["sources"].items())))
    print("keeper links (box lost or frozen -> another box):")
    for h in sorted(halves):
        for team in ("H", "A"):
            st = halves[h][team]["links"]
            links, undone = st["links"], st["undone"]
            print("  half %d %s: links %d (%d to a slot of the other nominal team); undone %d" % (
                h, team, len(links), sum(1 for s, _ in links if nominal[s] != team), len(undone)))
    counts = {}
    for h in rows:
        for r in rows[h]:
            counts[(h, r["verdict"])] = counts.get((h, r["verdict"]), 0) + 1
    verdicts = ["ok", "keeper", "acted as other team's keeper", "mostly frozen", "absent"]
    for h in sorted(rows):
        print("verdicts half %d: %s" % (h, ", ".join(
            "%s %d" % (v, counts.get((h, v), 0)) for v in verdicts)))
        flagged = [(r["slot"], r["verdict"]) for r in rows[h]
                   if r["verdict"] in ("acted as other team's keeper", "mostly frozen")]
        if flagged:
            print("  flagged slots: %s" % ", ".join("%d (%s)" % f for f in flagged))
    print("ball seen, no usable home slot: %d frames (half 1: %d, half 2: %d) in %d stretches, "
          "longest %d frames; %d frames in the first %d s of a half" % (
              gap["frames"], gap["halves"].get(1, 0), gap["halves"].get(2, 0), gap["stretches"],
              gap["longest"], gap["first_30s"], FIRST_SECONDS))
    print("  in those frames: mean %.1f home slots present, of which %.1f frozen" % (
        gap["home_present_mean"], gap["home_present_frozen_mean"]))
    print("malformed lines skipped: %d   run time %.1f s" % (malformed, time.time() - t0))

    if a.show:
        print("\n--show: per-frame rows for local debugging")
        show(halves, a.show, a.around)


if __name__ == "__main__":
    main()
