"""Step 7: the single reproduction command -- data bundle in, out/submission.csv out.

    python src/make_submission.py              # runs missing steps, merges, smooths, writes, validates
    python src/make_submission.py --rebuild    # reruns every step first (the reproduction command)

1. Pipeline: clean_table -> team_check -> ball_trips -> possession -> dead_ball, each run only if its
   out/ files are missing (all of them, in that order, with --rebuild).
2. Merge per video frame: outside the graded halves (timeutil) -> X. Inside: D from labels_dead.csv
   overrides everything; otherwise H/A from labels_raw.csv. Never X inside a graded half: a frame with
   no label holds the last H/A label; at the very start of a half, the first known H/A label after it.
3. Smoothing: an H or A run shorter than MIN_RUN_FRAMES lying between two runs of the same label
   (H | A-short | H) is merged into them. D runs are never smoothed or removed, nothing is merged
   across D, X is never changed. --no-smooth, --min-run.
4. Write out/submission.csv with starter/baseline.write_submission, validate that exact file with
   evaluation/validate_submission.py (non-zero exit on failure), copy it to
   evaluation/results/team_v1.csv (git-ignored; not a baseline_* name).
5. Write out/<TEAM_NAME>.csv (config.TEAM_NAME, --team overrides), a byte-identical copy: the file to
   upload. Validate it too (non-zero exit on failure); print both paths and their sha256.

Prints aggregates only.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = next((d for d in (HERE, os.path.dirname(HERE), os.getcwd())
             if os.path.isfile(os.path.join(d, "games.py"))), None)
if REPO is None:
    sys.exit("Can't find games.py. Run this from the repo root: python src/make_submission.py")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import games                                    # noqa: E402
from starter.baseline import write_submission   # noqa: E402
import timeutil as T                            # noqa: E402
import config as C                              # noqa: E402

OUT = os.path.join(REPO, "out")
# (script, the out/ files it writes) in pipeline order
STEPS = [
    ("clean_table.py", ["clean_table.csv"]),
    ("team_check.py", ["keepers.csv", "team_check.txt"]),
    ("ball_trips.py", ["ball_trips.csv", "touches.csv"]),
    ("possession.py", ["labels_raw.csv", "events.csv"]),
    ("dead_ball.py", ["labels_dead.csv"]),
]
RESULTS_COPY = os.path.join(REPO, "evaluation", "results", "team_v1.csv")


def run_pipeline(rebuild):
    """Run each step whose outputs are missing (every step with rebuild). Exit on a failure."""
    for script, outputs in STEPS:
        missing = [f for f in outputs if not os.path.isfile(os.path.join(OUT, f))]
        if not rebuild and not missing:
            print("  %-15s up to date" % script)
            continue
        t0 = time.time()
        res = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=REPO,
                             capture_output=True, text=True)
        if res.returncode != 0:
            sys.stderr.write(res.stdout[-2000:] + res.stderr[-2000:])
            sys.exit("step %s failed (exit %d)" % (script, res.returncode))
        print("  %-15s ran (%.1f s)" % (script, time.time() - t0))


def merge():
    """(labels for every video frame, per-half info). X outside the graded halves."""
    n = games.source_frames("tactical")
    labels = np.full(n, "X", dtype="<U1")
    raw = pd.read_csv(os.path.join(OUT, "labels_raw.csv"), keep_default_na=False,
                      dtype={"label": str}).set_index("video_frame")["label"]
    dead = pd.read_csv(os.path.join(OUT, "labels_dead.csv"), keep_default_na=False,
                       dtype={"label": str}).set_index("video_frame")["label"]
    info = {}
    for h, first, last in T.graded_blocks():
        vf = np.arange(first, last + 1)
        r = raw.reindex(vf).fillna("").to_numpy().astype(str)
        d = dead.reindex(vf).fillna("").to_numpy().astype(str)
        ha = np.where(np.isin(r, ["H", "A"]), r, "")
        # never X inside a half: hold the last H/A label; at the very start, the first one after it
        known = np.flatnonzero(ha != "")
        if len(known):
            idx = np.where(ha != "", np.arange(len(ha)), -1)
            idx = np.maximum.accumulate(idx)
            idx[idx < 0] = known[0]
            filled = ha[idx]
        else:
            filled = np.full(len(ha), "D")      # no possession information at all in this half
        info[h] = {"filled_frames": int((ha == "").sum())}
        labels[first:last + 1] = np.where(d == "D", "D", filled)
    return labels, info


def runs(seq):
    """[(start, stop, label)] constant runs of a 1-D array."""
    if not len(seq):
        return []
    br = np.flatnonzero(seq[1:] != seq[:-1]) + 1
    bounds = np.concatenate(([0], br, [len(seq)]))
    return [(int(bounds[i]), int(bounds[i + 1]), seq[bounds[i]]) for i in range(len(bounds) - 1)]


def smooth(labels, min_run):
    """Merge short H/A runs sandwiched between two runs of the same H/A label. Per half; D and X kept."""
    out = labels.copy()
    merged = {}
    for h, first, last in T.graded_blocks():
        seg = out[first:last + 1]
        rs = runs(seg)
        count = 0
        for k in range(1, len(rs) - 1):
            s, e, lab = rs[k]
            prev, nxt = rs[k - 1][2], rs[k + 1][2]
            if lab in ("H", "A") and e - s < min_run and prev == nxt and prev in ("H", "A") and prev != lab:
                seg[s:e] = prev
                count += 1
        out[first:last + 1] = seg
        merged[h] = count
    return out, merged


def validate(path):
    """Run evaluation/validate_submission.py on path, print its report, exit non-zero on failure."""
    res = subprocess.run([sys.executable, os.path.join(REPO, "evaluation", "validate_submission.py"),
                          path], cwd=REPO, capture_output=True, text=True)
    print("validation of %s:" % os.path.relpath(path, REPO))
    print("\n".join("  " + line for line in res.stdout.strip().splitlines()))
    if res.returncode != 0:
        sys.exit("validation FAILED for %s (exit %d)" % (path, res.returncode))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="rerun every step, in order")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--min-run", type=int, default=C.MIN_RUN_FRAMES,
                    help="frames; shorter H/A runs between two runs of the same label are merged")
    ap.add_argument("--out", default=os.path.join(OUT, "submission.csv"))
    ap.add_argument("--team", default=C.TEAM_NAME,
                    help="team name; the upload copy is out/<team>.csv (default: config.TEAM_NAME)")
    a = ap.parse_args()
    if not a.team or os.sep in a.team or "/" in a.team or a.team.startswith("."):
        sys.exit("--team must be a plain name (no path), e.g. --team %s" % C.TEAM_NAME)
    team_out = os.path.join(OUT, a.team + ".csv")
    if os.path.abspath(team_out) == os.path.abspath(a.out):
        sys.exit("--team gives the same file as --out (%s); pick another name" % a.out)

    t0 = time.time()
    print("pipeline:")
    run_pipeline(a.rebuild)

    labels, info = merge()
    merged = {h: 0 for h, _, _ in T.graded_blocks()}
    if not a.no_smooth:
        labels, merged = smooth(labels, a.min_run)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_submission(a.out, labels)
    validate(a.out)
    shutil.copyfile(a.out, RESULTS_COPY)
    os.makedirs(OUT, exist_ok=True)
    shutil.copyfile(a.out, team_out)
    validate(team_out)

    # -- aggregates only ------------------------------------------------------------------------
    print("wrote %s (%d rows) and copied it to %s" % (a.out, len(labels), os.path.relpath(RESULTS_COPY, REPO)))
    for h, first, last in T.graded_blocks():
        seg = labels[first:last + 1]
        rs = runs(seg)
        cnt = {k: sum(1 for _, _, l in rs if l == k) for k in ("H", "A", "D")}
        print("half %d: H %.1f%%  A %.1f%%  D %.1f%%  X %.1f%%; runs H %d, A %d, D %d; short runs merged "
              "by smoothing %d; frames with no step-5 label filled %d" % (
                  h, 100 * (seg == "H").mean(), 100 * (seg == "A").mean(), 100 * (seg == "D").mean(),
                  100 * (seg == "X").mean(), cnt["H"], cnt["A"], cnt["D"], merged[h],
                  info[h]["filled_frames"]))
    print("outside the graded halves: X on %d frames" % int((labels == "X").sum()))
    h_sub, h_team = sha256(a.out), sha256(team_out)
    print("sha256 %s  %s" % (h_sub, os.path.relpath(a.out, REPO)))
    print("sha256 %s  %s  (the file to upload)" % (h_team, os.path.relpath(team_out, REPO)))
    if h_sub != h_team:
        sys.exit("the two files differ")
    print("run time %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
