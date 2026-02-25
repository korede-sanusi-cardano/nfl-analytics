"""
NFL Stats Interface.

Uses nfl_data_py to pull historical stats, roster data, and play-by-play
from nflfastR. This is your primary source for actual NFL performance data.
"""

from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
from loguru import logger


class NFLStats:
    """Interface to historical NFL data via nfl_data_py."""

    FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]

    def __init__(self, cache_dir: Path = Path("data_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def get_weekly_data(self, seasons: list[int]) -> pd.DataFrame:
        """Get weekly fantasy scoring data for specified seasons."""
        cache_key = f"weekly_{'_'.join(map(str, seasons))}.parquet"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            logger.info(f"Loading weekly data from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Downloading weekly data for seasons: {seasons}")
        df = nfl.import_weekly_data(seasons)

        # Filter to fantasy-relevant positions and columns
        df = df[df["position"].isin(self.FANTASY_POSITIONS)].copy()

        df.to_parquet(cache_path)
        logger.info(f"Cached weekly data: {len(df)} rows")
        return df

    def get_seasonal_data(self, seasons: list[int]) -> pd.DataFrame:
        """Get season-level fantasy data."""
        cache_key = f"seasonal_{'_'.join(map(str, seasons))}.parquet"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        logger.info(f"Downloading seasonal data for: {seasons}")
        df = nfl.import_seasonal_data(seasons)
        df = df[df["position"].isin(self.FANTASY_POSITIONS)].copy()

        df.to_parquet(cache_path)
        return df

    def get_rosters(self, seasons: list[int]) -> pd.DataFrame:
        """Get end-of-season roster/demographic data (age, team, position)."""
        cache_key = f"seasonal_rosters_{'_'.join(map(str, seasons))}.parquet"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        logger.info(f"Downloading seasonal roster data for: {seasons}")
        df = nfl.import_seasonal_rosters(seasons)
        df = df[df["position"].isin(self.FANTASY_POSITIONS)].copy()

        df.to_parquet(cache_path)
        return df

    def get_weekly_rosters(self, seasons: list[int]) -> pd.DataFrame:
        """Get week-by-week roster snapshots (age, team, position)."""
        cache_key = f"weekly_rosters_{'_'.join(map(str, seasons))}.parquet"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        logger.info(f"Downloading weekly roster data for: {seasons}")
        df = nfl.import_weekly_rosters(seasons)
        df = df[df["position"].isin(self.FANTASY_POSITIONS)].copy()

        df.to_parquet(cache_path)
        return df

    def get_top_performers(
        self,
        season: int,
        position: str,
        scoring: str = "fantasy_points_ppr",
        top_n: int = 50,
    ) -> pd.DataFrame:
        """Get top fantasy performers at a position for a season."""
        weekly = self.get_weekly_data([season])
        rosters = self.get_rosters([season])

        player_totals = (
            weekly.groupby(["player_id", "player_name"])
            .agg(
                total_points=(scoring, "sum"),
                games_played=(scoring, "count"),
            )
            .reset_index()
        )
        player_totals["ppg"] = player_totals["total_points"] / player_totals["games_played"]

        # Add position
        roster_unique = rosters.drop_duplicates(subset=["player_id", "season"])[
            ["player_id", "position", "age", "team"]
        ].copy()

        merged = player_totals.merge(roster_unique, on="player_id", how="inner")
        merged = merged[merged["position"] == position]

        return (
            merged.sort_values("total_points", ascending=False).head(top_n).reset_index(drop=True)
        )

    def get_aging_curves(
        self,
        seasons: list[int],
        position: str,
        scoring: str = "fantasy_points_ppr",
        min_games: int = 8,
    ) -> pd.DataFrame:
        """
        Build empirical aging curves for a position.

        Returns average PPG by age, which you can use to calibrate
        your VORP model's decay assumptions.
        """
        weekly = self.get_weekly_data(seasons)
        rosters = self.get_rosters(seasons)

        # Aggregate to season level
        season_totals = (
            weekly.groupby(["player_id", "player_name", "season"])
            .agg(
                total_points=(scoring, "sum"),
                games_played=(scoring, "count"),
            )
            .reset_index()
        )
        season_totals["ppg"] = season_totals["total_points"] / season_totals["games_played"]

        # Filter minimum games
        season_totals = season_totals[season_totals["games_played"] >= min_games]

        # Add age
        roster_info = rosters.drop_duplicates(subset=["player_id", "season"])[
            ["player_id", "season", "position", "age"]
        ].copy()

        merged = season_totals.merge(roster_info, on=["player_id", "season"], how="inner")
        merged = merged[merged["position"] == position]

        # Average PPG by age
        aging = (
            merged.groupby("age")
            .agg(
                avg_ppg=("ppg", "mean"),
                median_ppg=("ppg", "median"),
                player_count=("player_id", "nunique"),
                sample_size=("ppg", "count"),
            )
            .reset_index()
        )

        return aging[aging["sample_size"] >= 10].sort_values("age")
