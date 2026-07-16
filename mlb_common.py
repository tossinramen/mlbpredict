"""Constants shared by build_history.py and mlb_predictor.py.

Keeping event classification and the FIP formula in one place guarantees
the live predictor computes features exactly the way the training rows in
mlb_history_2024_2025.csv were computed.
"""

FIP_CONSTANT = 3.15

# Outs credited to the pitcher per Statcast `events` value.
EVENT_OUTS = {
    "strikeout": 1, "strikeout_double_play": 2, "field_out": 1,
    "force_out": 1, "grounded_into_double_play": 2, "double_play": 2,
    "sac_fly": 1, "sac_bunt": 1, "fielders_choice_out": 1,
    "sac_fly_double_play": 2, "sac_bunt_double_play": 2, "triple_play": 3,
    "other_out": 1, "caught_stealing_2b": 1, "caught_stealing_3b": 1,
    "caught_stealing_home": 1, "pickoff_1b": 1, "pickoff_2b": 1,
    "pickoff_3b": 1,
}

HIT_EVENTS = {"single", "double", "triple", "home_run"}
BB_EVENTS = {"walk", "intent_walk", "intentional_walk"}
TB_MAP = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
NON_AB_EVENTS = BB_EVENTS | {
    "hit_by_pitch", "sac_fly", "sac_fly_double_play",
    "sac_bunt", "sac_bunt_double_play", "catcher_interf", "truncated_pa",
}


def fip(hr: float, bb: float, hbp: float, k: float, ip: float) -> float:
    """FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + constant."""
    return (13 * hr + 3 * (bb + hbp) - 2 * k) / ip + FIP_CONSTANT
