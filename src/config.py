"""Every threshold the pipeline uses, in one place, each with its football reason.

Values come from football sense and the Laws of the Game. Time thresholds are given in
seconds and in frames (25 fps); scripts use the frame versions. CLI flags may override them.
"""

# -- submission ------------------------------------------------------------------------------
TEAM_NAME = "InsightLens"     # name required by the organisers for the submission file

# -- units -----------------------------------------------------------------------------------
PITCH_L = 105.3   # pitch length in metres, as measured in games.py (the XML's 105 is nominal)
PITCH_W = 68.0    # pitch width in metres
FPS = 25          # tactical video and tracking frame rate


def frames(seconds):
    """Seconds -> whole frames at FPS."""
    return int(round(seconds * FPS))


# -- possession ------------------------------------------------------------------------------
CONTROL_RADIUS = 1.5          # m: a player can touch the ball from about a stride away
FAST_BALL = 8.0               # m/s: faster than a jog with the ball, so it is a pass/shot and the owner is held
# Touch detection. The tracker delivers an already smoothed, interpolated ball, so a kick or a trap
# appears as a speed ramp over about a quarter of a second, not as a jump between two frames.
TOUCH_SPEED_CHANGE = 3.0      # m/s: a kick or a trap changes ball speed suddenly; rolling friction does not
TOUCH_WINDOW_FRAMES = 6       # frames (0.24 s): the smoothed ball shows a kick/trap as a ramp this long
TOUCH_ANGLE = 30.0            # degrees: a pass or deflection turns the ball sharply; a rolling ball goes straight
TURN_CHORD_FRAMES = 3         # frames (0.12 s): direction is measured over chords this long, as the path is smoothed
TOUCH_MIN_MOVING_SPEED = 2.0  # m/s: below a walking-pace roll, a "turn" is position jitter, not a contact
TOUCH_MIN_GAP_FRAMES = 6      # frames (>= the window): one speed ramp is one contact; closer touches merge
SMOOTH_FRAMES = 1             # frames, 1 = off: the tracker's ball is already smooth; more only blurs touches
MAX_GAP_TIME = 1.0            # s: a missing ball longer than this may hide a touch, so it ends the trip
MAX_GAP_FRAMES = frames(MAX_GAP_TIME)        # = 25
THROW_RISE_TIME = 0.48        # s: an arm swing builds speed clearly slower than a kick's ~0.24 s ramp (2x)
THROW_RISE_FRAMES = frames(THROW_RISE_TIME)  # = 12
TOUCH_ATTRIB_FRAMES = 3       # frames each side of the onset searched (only if nobody is at the ball at onset)
TOUCH_ATTRIB_RADIUS = 2.0     # m, for finding the toucher (CONTROL_RADIUS stays for `close`): a foot reaches the
                              # ball up to ~1.5 m from the body's tracked point, plus ~0.5 m tracking error per
                              # box; 2.5 m starts reaching a second player
ONSET_MAX_TIME = 1.0          # s: a kick's ramp starts, or a trap's slowdown ends, within ~1 s of its peak change
ONSET_MAX_FRAMES = frames(ONSET_MAX_TIME)    # = 25
# Stop-and-go: a ball on grass only slows down by itself; it never speeds up or turns on its own.
STOP_SPEED = 1.5              # m/s: slower than a walking-pace roll -- the ball has been stopped or nearly
STOP_GO_RISE = 2.0            # m/s: a clear new push, well above the tracker's speed noise (p99 ~1 m/s/frame)
STOP_GO_TIME = 0.48           # s: the push follows the stop within about half a second
STOP_GO_FRAMES = frames(STOP_GO_TIME)        # = 12
DRIBBLE_MAX_TRIP_TIME = 1.0   # s: a dribbling player touches the ball at least about once a second
DRIBBLE_MAX_TRIP_FRAMES = frames(DRIBBLE_MAX_TRIP_TIME)  # = 25
# Box distribution (step 5): a player who holds the ball inside a penalty area and then plays it long,
# away from that goal, belongs to the team defending that goal (a keeper distributing, a defender
# clearing); an attacker in the opponent's box does not hold the ball and then play it back 20 m.
DIST_HOLD_SPEED = 3.0         # m/s: slower than a jog -- the ball is held in hands or at the feet
DIST_HOLD_TIME = 1.5          # s: a keeper's or defender's composed possession before distributing
DIST_HOLD_FRAMES = frames(DIST_HOLD_TIME)          # = 38
DIST_HOLD_MAX_GAP_TIME = 0.2  # s: brief ball dropouts (hands, bodies) don't end the hold
DIST_HOLD_MAX_GAP_FRAMES = frames(DIST_HOLD_MAX_GAP_TIME)   # = 5
DIST_MIN_TRIP_M = 20.0        # m: a distribution or clearance, not a short pass in the box
DIST_GAIN_M = 15.0            # m further from that goal's centre: played away from the goal it defends
DIST_MAX_FLIGHT_S = 4.0       # s: a long throw or kick has landed with someone by then; measure there at the latest
DIST_MAX_FLIGHT_FRAMES = frames(DIST_MAX_FLIGHT_S)   # = 100
MIN_OWN_TIME = 0.4            # s: keeping the ball this long is control (own), shorter is a deflection
MIN_OWN_FRAMES = frames(MIN_OWN_TIME)   # = 10

# -- smoothing -------------------------------------------------------------------------------
MIN_RUN = 0.4                 # s: a real possession lasts longer; shorter label runs are noise
MIN_RUN_FRAMES = frames(MIN_RUN)        # = 10

# -- dead ball -------------------------------------------------------------------------------
OUT_MARGIN = 0.5              # m beyond a line before the ball counts as out: tracking noise
FAR_OUT_MARGIN = 1.5          # m on the far touchline (y = 0): positions there are less accurate

MIN_DEAD_TIME = 1.0           # s: a restart (throw-in, goal kick, free kick) takes longer; shorter D would only
                              # split a real possession in two (a spellF1 penalty)
MIN_DEAD_FRAMES = frames(MIN_DEAD_TIME)      # = 25
KICKOFF_SPOT_RADIUS_M = 2.0   # m: the ball is placed on the centre spot; allows for ball-position error
KICKOFF_SEARCH_S = 300.0      # s: a half kicks off within minutes of grading starting (FAQ: ~2 min in half 2)
KICKOFF_SEARCH_FRAMES = frames(KICKOFF_SEARCH_S)   # = 7500
# Kick-off line-up: before every kick-off all players stand in their own half,
# and the tracker sees players even when it does not see the ball.
KICKOFF_OWN_HALF_SHARE = 0.9  # at a kick-off every player is in his own half; 0.9 allows a few tracking errors
KICKOFF_MIN_PLAYERS = 16      # tracked players needed to judge the line-up (most of the 22); fewer -> the
                              # frame continues the previous state, so dropped boxes don't cut a line-up
KICKOFF_LINEUP_TIME = 2.0     # s: players hold their kick-off positions for seconds before the whistle
KICKOFF_LINEUP_FRAMES = frames(KICKOFF_LINEUP_TIME)   # = 50
HALFWAY_TOL_M = 1.0           # m: tracking noise around the halfway line
AFTER_GOAL_MAX_S = 90.0       # s: a goal-to-restart takes ~40-90 s (celebration, walk back); the after-goal D
                              # never starts earlier than this before the line-up
AFTER_GOAL_MAX_FRAMES = frames(AFTER_GOAL_MAX_S)   # = 2250
GOAL_AREA_DEPTH_M = 5.5       # m: six-yard box depth (Laws of the Game)
GOAL_AREA_WIDTH_M = 18.32     # m: six-yard box width (Laws of the Game)
FAR_OUT_SUSTAIN_TIME = 1.0    # s: a far-side "out" must stay beyond the line this long; a high ball on the far
                              # side can project beyond the touchline for a moment while still in play
FAR_OUT_SUSTAIN_FRAMES = frames(FAR_OUT_SUSTAIN_TIME)   # = 25
THROWIN_MIN_S = 3.0           # s: a throw-in can't be taken faster -- someone has to fetch the ball and set up
THROWIN_MIN_FRAMES = frames(THROWIN_MIN_S)              # = 75
PENALTY_AREA_DEPTH_M = 16.5   # m (Laws of the Game). After a goal the ball is in the net or being carried
PENALTY_AREA_WIDTH_M = 40.32  # back; D starts when it was last seen in a penalty area (the shot / the net)

# -- step 6 checks (report only, never change a label) -----------------------------------------
FAR_OUT_CHECK_MIN_S = 3.0     # s: a real throw-in stoppage lasts longer; a shorter far-side D may be tracking error
FAR_OUT_CHECK_MIN_DEPTH_M = 1.0   # m: a ball barely "out" on the inaccurate far side may never have left the pitch
THROWIN_NEAR_LINE_M = 5.0     # m: a throw-in restarts with the ball close to the touchline it left by
TAG_MATCH_S = 10.0            # s: Smart Tagging windows open ~3.5 s early and last ~5 s

# -- step 6 restart-spot rule: throw-ins and corners the tracker didn't see go out -----------------
# (needs a Smart Tagging window of the same Type. Goal kicks not applied: only 3
# of 12 matched a tag in the measurement. A "ball reappears still" condition was dropped: the tracker
# picks the ball up again once it is played -- median 11.7 m/s on reappearing.)
RESTART_GAP_FRAMES = 25       # >= 1 s without the ball: long enough to hide it going out and being fetched
RESTART_SPOT_M = 2.0          # m: a throw-in is taken at the touchline
RESTART_CORNER_M = 3.0        # m: a corner is taken from the corner arc (1 m) plus ball-position error
RESTART_MAX_DEAD_S = 20.0     # s: fetching the ball and setting up a throw-in or corner takes ~5-20 s; a
                              # longer gap holds real play we didn't see
RESTART_MAX_DEAD_FRAMES = frames(RESTART_MAX_DEAD_S)   # = 500

# -- data cleaning (step 1) ------------------------------------------------------------------
FROZEN_TIME = 1.0             # s: no real player stands perfectly still this long, so it is a stuck track
FROZEN_FRAMES = frames(FROZEN_TIME)     # = 25
FROZEN_MAX_PATH_M = 0.15      # m of total path in FROZEN_FRAMES: an abandoned box only drifts...
FROZEN_TWIN_RADIUS_M = 3.0    # m: ...and the real person carries on in a new box beside it. A still box
                              # counts as frozen only with such a live twin; a lone still player (a keeper
                              # while play is far away) is not a ghost. Bit-identical positions always freeze.
SANE_BALL_SPEED = 40.0        # m/s: above the hardest shots (~35 m/s); faster readings are tracking glitches

# -- keeper detection (step 3, team_check.py) ------------------------------------------------
KEEPER_ZONE_M = 16.5          # m from the goal centre (penalty-box depth): a keeper rarely leaves his box
KEEPER_WINDOW_S = 10.0        # s: a keeper holds his post for whole phases of play, not for moments
KEEPER_WINDOW_FRAMES = frames(KEEPER_WINDOW_S)   # = 250
KEEPER_PERSIST = 0.70         # share of that window: an attacker drifting deep at a corner is there far less
KEEPER_PERSIST_FRAMES = int(round(KEEPER_PERSIST * KEEPER_WINDOW_FRAMES))   # = 175
MOSTLY_FROZEN_SHARE = 0.5     # a slot frozen for over half its presence is a stuck track, not a player
KEEPER_KEEP_ZONE_M = 30.0     # m from goal centre: an established keeper rarely goes beyond his box area
KEEPER_RUN_SPEED = 7.0        # m/s: a keeper moving with the ball (run-up to a kick or throw) is no faster
KEEPER_LINK_REACH_M = 1.0     # m: plus box-position noise, so a box right beside the lost one always qualifies
KEEPER_LINK_MAX_TIME = 2.0    # s: the tracker re-boxes a lost keeper quickly; later, a nearby box is someone else
KEEPER_LINK_MAX_FRAMES = frames(KEEPER_LINK_MAX_TIME)   # = 50
KEEPER_LINK_UNIQUE_M = 2.0    # m: link to the nearest live box only if no other live box is this close to it
KEEPER_LINK_UNDO_TIME = 5.0   # s: a linked box that leaves the keep zone this soon was an outfield player
KEEPER_LINK_UNDO_FRAMES = frames(KEEPER_LINK_UNDO_TIME)  # = 125
