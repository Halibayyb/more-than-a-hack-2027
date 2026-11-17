"""Step 2: frame <-> time helpers for the whole pipeline. Importable module, no CLI.

Every frame maths here wraps games.py. Naming:
    vf  = video frame (0-based, the submission row)
    fid = frame_id    (1-based, first field of a tracking line; vf = fid - 1, see
                       starter/load_atd.video_frame_of)

    half(vf)            1, 2, or None outside the graded halves
    is_graded(vf)       True inside a graded half
    graded_blocks()     [(half, first_vf, last_vf), ...]
    video_time(vf)      "HH:MM:SS.cc" since the first frame of the video
    match_seconds(vf)   seconds on the match clock, None outside the graded halves
    match_time(vf)      "HH:MM:SS.cc" on the match clock, "-----------" outside the graded halves
    clock(seconds)      "HH:MM:SS.cc"; "-----------" for a negative value

match_time is for display only. It starts at 00:00 / 45:00 on the first graded frame of each half.
The 2nd half is graded from before its kick-off, so there it runs ahead of the real match clock.

    python src/timeutil.py          # self-check, needs no match data
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = next((d for d in (HERE, os.path.dirname(HERE), os.getcwd())
             if os.path.isfile(os.path.join(d, "games.py"))), None)
if REPO is None:
    raise ImportError("Can't find games.py; run from the repo root.")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import games  # noqa: E402

GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
# Match clock at the first graded frame of each half: 0' and 45'.
HALF_CLOCK_START_S = {1: 0.0, 2: 45 * 60.0}
NO_TIME = "-" * 11


def graded_blocks():
    """[(half, first_vf, last_vf)] in play order, from games.half_video_blocks()."""
    return games.half_video_blocks(GAME)


def half(vf):
    """1 or 2 inside a graded half, None for pre-match, halftime and after the final whistle."""
    return games.half_of_video_frame(GAME, vf)


def is_graded(vf):
    return half(vf) is not None


def clock(seconds):
    """HH:MM:SS.cc, e.g. 00:07:29.04 (cc = hundredths of a second). Dashes for a negative value."""
    if seconds < 0:
        return NO_TIME
    return "%02d:%02d:%05.2f" % (seconds // 3600, seconds % 3600 // 60, seconds % 60)


def video_time(vf):
    """Time since the first frame of the tactical video."""
    return clock(vf / FPS)


_HALF_FIRST = {h: first for h, first, _ in graded_blocks()}


def match_seconds(vf):
    """Seconds on the match clock (display only), or None outside the graded halves."""
    h = half(vf)
    if h is None:
        return None
    return HALF_CLOCK_START_S[h] + (vf - _HALF_FIRST[h]) / FPS


def match_time(vf):
    """Match clock as HH:MM:SS.cc (display only); dashes outside the graded halves."""
    seconds = match_seconds(vf)
    return NO_TIME if seconds is None else clock(seconds)


def _self_check():
    """Checks that need no match data. Prints PASS/FAIL per check; returns True if all pass."""
    blocks = graded_blocks()
    (h1, f1, l1), (h2, f2, l2) = blocks
    checks = [
        ("graded blocks are halves 1 and 2, in order, not overlapping",
         len(blocks) == 2 and (h1, h2) == (1, 2) and l1 < f2),
        ("block lengths 79467 and 79390", (l1 - f1 + 1, l2 - f2 + 1) == (79467, 79390)),
        ('match_time(11185) == "00:00:00.00"', match_time(11185) == "00:00:00.00"),
        ('match_time(110808) == "00:45:00.00"', match_time(110808) == "00:45:00.00"),
        ('video_time(11185) == "00:07:27.40"', video_time(11185) == "00:07:27.40"),
        ("half() at block edges and outside",
         (half(f1), half(l1), half(f2), half(l2), half(f1 - 1), half(l1 + 1), half(l2 + 1))
         == (1, 1, 2, 2, None, None, None)),
        ("is_graded() agrees with half()", is_graded(f1) and not is_graded(l1 + 1)),
        ("match_time() outside the halves is dashes", match_time(f1 - 1) == NO_TIME),
    ]
    print("graded blocks: %s" % blocks)
    for name, ok in checks:
        print("  %s  %s" % ("PASS" if ok else "FAIL", name))
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    sys.exit(0 if _self_check() else 1)
