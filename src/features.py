import pandas as pd

def add_basic_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple football-derived features:
    - goal difference
    - is home favourite (based on odds)
    - implied probabilities from odds
    """
    out = df.copy()

    # Full-time goal difference
    out["GoalDiff"] = out["FTHG"] - out["FTAG"]

    # Odds — lower value means favourite
    out["HomeFav"] = (out["B365H"] < out["B365A"]).astype(int)

    # Convert odds to implied probabilities
    out["p_home"] = 1 / out["B365H"]
    out["p_draw"] = 1 / out["B365D"]
    out["p_away"] = 1 / out["B365A"]

    # Normalize to sum to 1 (remove bookmaker margin)
    prob_sum = out[["p_home", "p_draw", "p_away"]].sum(axis=1)
    out["p_home"] /= prob_sum
    out["p_draw"] /= prob_sum
    out["p_away"] /= prob_sum

    return out

"The below function computes avg goals scored, conceded, and average points per match for the last 5 games for each team."

def add_team_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Adds rolling-form features for each team based on the last N matches.
    Features are computed separately for home and away matches.
    """
    out = df.copy()

    # Points for result
    result_to_points = {0: 3, 1: 1, 2: 0}  # H,D,A from home perspective
    out["HomePoints"] = out["Result"].map({0: 3, 1: 1, 2: 0})
    out["AwayPoints"] = out["Result"].map({0: 0, 1: 1, 2: 3})

    # Build long-form table to compute rolling features per team
    home_df = out[["Date", "HomeTeam", "FTHG", "FTAG", "HomePoints"]].rename(
        columns={"HomeTeam": "Team", "FTHG": "GF", "FTAG": "GA", "HomePoints": "Points"}
    )
    away_df = out[["Date", "AwayTeam", "FTAG", "FTHG", "AwayPoints"]].rename(
        columns={"AwayTeam": "Team", "FTAG": "GF", "FTHG": "GA", "AwayPoints": "Points"}
    )

    long_df = pd.concat([home_df, away_df], ignore_index=True).sort_values("Date")

    # Rolling averages
    long_df["GF_roll"] = long_df.groupby("Team")["GF"].rolling(window).mean().reset_index(level=0, drop=True)
    long_df["GA_roll"] = long_df.groupby("Team")["GA"].rolling(window).mean().reset_index(level=0, drop=True)
    long_df["PTS_roll"] = long_df.groupby("Team")["Points"].rolling(window).mean().reset_index(level=0, drop=True)

    # Merge back to match rows
    out = out.merge(
        long_df[["Date", "Team", "GF_roll", "GA_roll", "PTS_roll"]],
        left_on=["Date", "HomeTeam"],
        right_on=["Date", "Team"],
        how="left"
    ).rename(columns={
        "GF_roll": "Home_GF_roll",
        "GA_roll": "Home_GA_roll",
        "PTS_roll": "Home_PTS_roll"
    }).drop(columns=["Team"])

    out = out.merge(
        long_df[["Date", "Team", "GF_roll", "GA_roll", "PTS_roll"]],
        left_on=["Date", "AwayTeam"],
        right_on=["Date", "Team"],
        how="left"
    ).rename(columns={
        "GF_roll": "Away_GF_roll",
        "GA_roll": "Away_GA_roll",
        "PTS_roll": "Away_PTS_roll"
    }).drop(columns=["Team"])

    return out
