import pandas as pd
import numpy as np

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

def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds relative-strength and balance features comparing
    home and away rolling form statistics.
    """
    out = df.copy()

    # Relative strength (home minus away)
    out["PTS_gap"] = out["Home_PTS_roll"] - out["Away_PTS_roll"]
    out["GF_gap"]  = out["Home_GF_roll"]  - out["Away_GF_roll"]
    out["GA_gap"]  = out["Away_GA_roll"]  - out["Home_GA_roll"]  # defensive advantage

    # Balance indicators (smaller -> more evenly matched -> draw-like)
    out["Form_balance_PTS"] = (out["Home_PTS_roll"] - out["Away_PTS_roll"]).abs()
    out["Form_balance_GF"]  = (out["Home_GF_roll"]  - out["Away_GF_roll"]).abs()
    out["Form_balance_GA"]  = (out["Home_GA_roll"]  - out["Away_GA_roll"]).abs()

    return out


def add_team_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds season-long team strength features based on all matches
    played so far in that season (home + away combined).

    For each (Season, Team) it computes expanding (cumulative) averages
    of goals scored, goals conceded, and points per match, then attaches
    them as separate features for the home and away team in each match.
    """
    out = df.copy()

    # Make sure we have season + points info.
    # Result: 0 = home win, 1 = draw, 2 = away win
    home_points_map = {0: 3, 1: 1, 2: 0}
    away_points_map = {0: 0, 1: 1, 2: 3}

    out["HomePoints"] = out["Result"].map(home_points_map)
    out["AwayPoints"] = out["Result"].map(away_points_map)

    # Build long table: one row per (team, match) with season + date
    home_df = out[["SeasonFile", "Date", "HomeTeam", "FTHG", "FTAG", "HomePoints"]].rename(
        columns={
            "HomeTeam": "Team",
            "FTHG": "GF",
            "FTAG": "GA",
            "HomePoints": "Points",
        }
    )

    away_df = out[["SeasonFile", "Date", "AwayTeam", "FTAG", "FTHG", "AwayPoints"]].rename(
        columns={
            "AwayTeam": "Team",
            "FTAG": "GF",
            "FTHG": "GA",
            "AwayPoints": "Points",
        }
    )

    long_df = pd.concat([home_df, away_df], ignore_index=True)
    long_df = long_df.sort_values(["SeasonFile", "Date"])

    # Group by (season, team) and compute expanding averages
    group = long_df.groupby(["SeasonFile", "Team"])

    long_df["Season_GF_avg"] = (
        group["GF"].expanding().mean().reset_index(level=[0, 1], drop=True)
    )
    long_df["Season_GA_avg"] = (
        group["GA"].expanding().mean().reset_index(level=[0, 1], drop=True)
    )
    long_df["Season_PTS_avg"] = (
        group["Points"].expanding().mean().reset_index(level=[0, 1], drop=True)
    )

    # Merge home team strength back onto match rows
    out = out.merge(
        long_df[
            ["SeasonFile", "Date", "Team",
             "Season_GF_avg", "Season_GA_avg", "Season_PTS_avg"]
        ],
        left_on=["SeasonFile", "Date", "HomeTeam"],
        right_on=["SeasonFile", "Date", "Team"],
        how="left",
    ).rename(
        columns={
            "Season_GF_avg": "Home_Season_GF_avg",
            "Season_GA_avg": "Home_Season_GA_avg",
            "Season_PTS_avg": "Home_Season_PTS_avg",
        }
    ).drop(columns=["Team"])

    # Merge away team strength
    out = out.merge(
        long_df[
            ["SeasonFile", "Date", "Team",
             "Season_GF_avg", "Season_GA_avg", "Season_PTS_avg"]
        ],
        left_on=["SeasonFile", "Date", "AwayTeam"],
        right_on=["SeasonFile", "Date", "Team"],
        how="left",
    ).rename(
        columns={
            "Season_GF_avg": "Away_Season_GF_avg",
            "Season_GA_avg": "Away_Season_GA_avg",
            "Season_PTS_avg": "Away_Season_PTS_avg",
        }
    ).drop(columns=["Team"])

    # Gap features based on season-long strength
    out["Season_PTS_gap"] = out["Home_Season_PTS_avg"] - out["Away_Season_PTS_avg"]
    out["Season_GF_gap"]  = out["Home_Season_GF_avg"]  - out["Away_Season_GF_avg"]
    out["Season_GA_gap"]  = out["Away_Season_GA_avg"]  - out["Home_Season_GA_avg"]
    
    # --- Home-only and Away-only rolling form splits ---

    # Home team performance when playing AT HOME
    home_home_matches = (
        out[out["HomeTeam"] == out["HomeTeam"]]
    )

    # Away team performance when playing AWAY
    away_away_matches = (
        out[out["AwayTeam"] == out["AwayTeam"]]
    )

    # Rolling home-at-home stats for HomeTeam
    out["Home_Home_PTS_roll"] = (
        out.groupby("HomeTeam")["HomePoints"]
        .rolling(window=5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    out["Home_Home_GF_roll"] = (
        out.groupby("HomeTeam")["FTHG"]
        .rolling(window=5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    out["Home_Home_GA_roll"] = (
        out.groupby("HomeTeam")["FTAG"]
        .rolling(window=5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Rolling away-away stats for AwayTeam
    out["Away_Away_PTS_roll"] = (
        out.groupby("AwayTeam")["AwayPoints"]
        .rolling(window=5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    out["Away_Away_GF_roll"] = (
        out.groupby("AwayTeam")["FTAG"]
        .rolling(window=5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    out["Away_Away_GA_roll"] = (
        out.groupby("AwayTeam")["FTHG"]
        .rolling(window=5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    out["PTS_gap_split"] = (
        out["Home_Home_PTS_roll"] - out["Away_Away_PTS_roll"]
    )

    out["GF_gap_split"] = (
        out["Home_Home_GF_roll"] - out["Away_Away_GF_roll"]
    )

    out["GA_gap_split"] = (
        out["Away_Away_GA_roll"] - out["Home_Home_GA_roll"]
    )


    return out

def add_elo_features(df, k_factor=20, start_rating=1500):
    ratings = {}
    elo_home = []
    elo_away = []
    elo_diff = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        ratings.setdefault(home, start_rating)
        ratings.setdefault(away, start_rating)

        Rh = ratings[home]
        Ra = ratings[away]

        exp_home = 1 / (1 + 10 ** ((Ra - Rh) / 400))
        exp_away = 1 - exp_home

        elo_home.append(Rh)
        elo_away.append(Ra)
        elo_diff.append(Rh - Ra)

        # Update ratings AFTER match
        result = row["Result"]
        if result == 0:   # home win
            Sh, Sa = 1, 0
        elif result == 1: # draw
            Sh, Sa = 0.5, 0.5
        else:             # away win
            Sh, Sa = 0, 1

        ratings[home] = Rh + k_factor * (Sh - exp_home)
        ratings[away] = Ra + k_factor * (Sa - exp_away)

    df["ELO_Home"] = elo_home
    df["ELO_Away"] = elo_away
    df["ELO_Diff"] = elo_diff

    return df
