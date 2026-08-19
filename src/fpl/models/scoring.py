"""Official FPL scoring weights used by the structural xPts model."""

GOAL_POINTS = {"GKP": 6, "GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_POINTS = {"GKP": 4, "GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3
YELLOW_POINTS = -1
RED_POINTS = -3
OWN_GOAL_POINTS = -2
PENALTY_MISS_POINTS = -2
PENALTY_SAVE_POINTS = 5
SAVE_BUNDLE = 3  # 1 point per 3 saves
GC_BUNDLE = 2  # -1 per 2 goals conceded (GKP/DEF)
