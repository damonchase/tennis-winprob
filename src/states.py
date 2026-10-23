import pandas as pd


def state_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Validate tennis score states and return a DataFrame of boolean violation flags."""
    v = pd.DataFrame(index=df.index)

    p1_sets = df["p1_sets"]
    p2_sets = df["p2_sets"]
    set_no = df["set_no"]
    p1_games = df["p1_games"]
    p2_games = df["p2_games"]
    p1_pts = df["p1_pts"]
    p2_pts = df["p2_pts"]
    is_tiebreak = df["is_tiebreak"]
    server = df["server"]

    # 1. sets_too_many: player has won 3+ sets in best-of-five when set_no > 3
    v["sets_too_many"] = (
        ((p1_sets > 2) & (set_no > 3)) | 
        ((p2_sets > 2) & (set_no > 3)) | 
        (p1_sets > 3) | 
        (p2_sets > 3)
    )

    # 2. sets_dont_add_up: sets won cannot be >= current set number
    v["sets_dont_add_up"] = (p1_sets >= set_no) | (p2_sets >= set_no) | (p1_sets + p2_sets >= set_no)

    # 3. set_already_over: active point in a set that has already been won
    normal_set_won = (is_tiebreak == 0) & (
        ((p1_games >= 6) & (p1_games - p2_games >= 2)) |
        ((p2_games >= 6) & (p2_games - p1_games >= 2)) |
        ((p1_games == 7) & (p2_games == 5)) |
        ((p2_games == 7) & (p1_games == 5))
    )
    tb_set_won = (is_tiebreak == 1) & (
        ((p1_pts >= 7) & (p1_pts - p2_pts >= 2)) |
        ((p2_pts >= 7) & (p2_pts - p1_pts >= 2))
    )
    v["set_already_over"] = normal_set_won | tb_set_won

    # 4. bad_game_points: game already won by point count in a normal game
    game_won_by_pts = (is_tiebreak == 0) & (
        ((p1_pts >= 4) & (p1_pts - p2_pts >= 2)) |
        ((p2_pts >= 4) & (p2_pts - p1_pts >= 2))
    )
    v["bad_game_points"] = game_won_by_pts | (p1_pts < 0) | (p2_pts < 0)

    # 5. bad_tiebreak: tiebreak active when games are not 6-6
    v["bad_tiebreak"] = (is_tiebreak == 1) & ((p1_games != 6) | (p2_games != 6))

    # 6. bad_server: server must be 1 or 2
    v["bad_server"] = (server != 1) & (server != 2)

    return v


def invalid_state_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask indicating rows with any state violation."""
    v = state_violations(df)
    return v.any(axis=1)