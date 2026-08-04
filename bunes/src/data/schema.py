"""The single unified trajectory schema.

Every data source (NGSIM, highD, synthetic) is converted into a dataframe with
exactly these columns and units. Everything downstream — windowing, the model,
the Phase 2 snapper, the Phase 3 IDM loss — depends only on this contract.

Coordinate convention (chosen once, used everywhere):
    x : longitudinal, metres, increasing in the direction of travel
    y : lateral,      metres, increasing to the left of the direction of travel
    heading : radians, atan2(dy, dx), so ~0 for a vehicle tracking its lane

This means NGSIM's Local_Y maps to x and Local_X maps to y, and highD vehicles
travelling in the -x direction are mirrored so all traffic flows +x.
"""

from __future__ import annotations

# --- column names -----------------------------------------------------------
VEHICLE_ID = "vehicle_id"      # globally unique across recordings
FRAME = "frame"                # integer frame index, uniform sampling
TIME = "t"                     # seconds
X = "x"                        # longitudinal position [m]
Y = "y"                        # lateral position [m]
VX = "vx"                      # longitudinal velocity [m/s]
VY = "vy"                      # lateral velocity [m/s]
SPEED = "speed"                # |v| [m/s]
ACCEL = "accel"                # longitudinal acceleration [m/s^2]
HEADING = "heading"            # [rad]
LANE_ID = "lane_id"            # integer lane index
LENGTH = "length"              # vehicle length [m]
WIDTH = "width"                # vehicle width [m]
LEADER_ID = "leader_id"        # id of preceding vehicle, -1 if none
GAP = "gap"                    # bumper-to-bumper space headway [m], nan if none
LEADER_SPEED = "leader_speed"  # preceding vehicle speed [m/s], nan if none

UNIFIED_COLUMNS = [
    VEHICLE_ID, FRAME, TIME, X, Y, VX, VY, SPEED, ACCEL, HEADING,
    LANE_ID, LENGTH, WIDTH, LEADER_ID, GAP, LEADER_SPEED,
]

# `leader_id`, `gap` and `leader_speed` are unused in Phase 1 but are carried
# through the pipeline now because the Phase 3 IDM loss needs them, and
# re-running preprocessing over full NGSIM later is expensive.

# --- per-timestep model input features (agent frame) ------------------------
# Order matters: it defines the channel layout of the encoder input tensor and
# of the fitted feature scaler.
FEATURE_NAMES = [
    "rel_x",      # position relative to the last observed point, rotated
    "rel_y",
    "delta_x",    # per-step displacement, rotated
    "delta_y",
    "speed",
    "accel",
    "sin_heading",  # heading relative to the agent frame
    "cos_heading",
    "lane_offset",  # lateral distance to own lane centre (0 if unknown)
]
NUM_FEATURES = len(FEATURE_NAMES)

# Unit conversions used by the raw parsers.
FEET_TO_M = 0.3048
