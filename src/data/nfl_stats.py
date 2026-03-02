"""
NFL Stats Interface.

Uses nfl_data_py to pull historical stats, roster data, and play-by-play
from nflfastR. This is your primary source for actual NFL performance data.
Falls back to the Sleeper API when nflverse data is unavailable (e.g. the
current season hasn't been published yet).
"""

import datetime
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
        """Get weekly fantasy scoring data for specified seasons.

        Tries nflverse first; falls back to the Sleeper API if that season's
        data hasn't been published yet (e.g. the most recent completed season).
        """
        combined_cache = self.cache_dir / f"weekly_{'_'.join(map(str, seasons))}.parquet"
        if combined_cache.exists():
            logger.info(f"Loading weekly data from cache: {combined_cache}")
            return pd.read_parquet(combined_cache)

        frames: list[pd.DataFrame] = []
        for season in seasons:
            season_cache = self.cache_dir / f"weekly_{season}.parquet"
            if season_cache.exists():
                frames.append(pd.read_parquet(season_cache))
                continue

            try:
                logger.info(f"Downloading weekly data for {season} from nflverse")
                df = nfl.import_weekly_data([season])
                df = df[df["position"].isin(self.FANTASY_POSITIONS)].copy()
                logger.info(f"nflverse: {len(df)} rows for {season}")
            except Exception as exc:
                logger.warning(
                    f"nflverse unavailable for {season} ({exc}); "
                    "falling back to Sleeper API"
                )
                df = self._build_weekly_from_sleeper(season)
                if df.empty:
                    raise RuntimeError(
                        f"Could not load weekly data for {season} from nflverse "
                        f"or Sleeper. The season data may not be available yet."
                    ) from exc
                logger.info(f"Sleeper fallback: {len(df)} rows for {season}")

            df.to_parquet(season_cache)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(combined_cache)
        logger.info(f"Cached combined weekly data: {len(combined)} rows")
        return combined

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
        """Get end-of-season roster/demographic data (age, team, position).

        Tries nflverse first; falls back to the Sleeper API player database
        when the season hasn't been published on nflverse yet.
        """
        combined_cache = self.cache_dir / f"seasonal_rosters_{'_'.join(map(str, seasons))}.parquet"
        if combined_cache.exists():
            return pd.read_parquet(combined_cache)

        frames: list[pd.DataFrame] = []
        for season in seasons:
            season_cache = self.cache_dir / f"seasonal_rosters_{season}.parquet"
            if season_cache.exists():
                frames.append(pd.read_parquet(season_cache))
                continue

            try:
                logger.info(f"Downloading roster data for {season} from nflverse")
                df = nfl.import_seasonal_rosters([season])
                df = df[df["position"].isin(self.FANTASY_POSITIONS)].copy()
            except Exception as exc:
                logger.warning(
                    f"nflverse rosters unavailable for {season} ({exc}); "
                    "falling back to Sleeper API"
                )
                df = self._build_rosters_from_sleeper(season)
                if df.empty:
                    raise RuntimeError(
                        f"Could not load roster data for {season} from nflverse "
                        f"or Sleeper. The season data may not be available yet."
                    ) from exc

            df.to_parquet(season_cache)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(combined_cache)
        return combined

    # ------------------------------------------------------------------
    # Sleeper fallback builders
    # ------------------------------------------------------------------

    def _build_weekly_from_sleeper(self, season: int) -> pd.DataFrame:
        """Fetch per-week player stats from Sleeper and return them in the
        same schema as nflverse weekly data (player_id, player_name, season,
        fantasy_points_ppr, fantasy_points, position)."""
        from src.data.sleeper_client import SleeperClient

        client = SleeperClient()
        all_players = client.get_all_players()

        rows: list[dict] = []
        for week in range(1, 19):
            try:
                week_stats = client.get_weekly_stats(season, week)
            except Exception as exc:
                logger.info(f"Sleeper: no data for {season} week {week} ({exc})")
                break

            if not week_stats:
                break

            for player_id, stats in week_stats.items():
                pts_ppr = stats.get("pts_ppr")
                if pts_ppr is None:
                    continue

                info = all_players.get(str(player_id), {})
                pos = info.get("position")
                if pos not in self.FANTASY_POSITIONS:
                    continue

                first = info.get("first_name", "")
                last = info.get("last_name", "")
                full_name = f"{first} {last}".strip() or info.get("full_name", "Unknown")

                rows.append({
                    "player_id": str(player_id),
                    "player_name": full_name,
                    "position": pos,
                    "season": season,
                    "week": week,
                    "fantasy_points_ppr": float(pts_ppr),
                    "fantasy_points": float(stats.get("pts_std", 0.0)),
                })

        return pd.DataFrame(rows)

    def _build_rosters_from_sleeper(self, season: int) -> pd.DataFrame:
        """Build a roster DataFrame from the Sleeper player database, matching
        the nflverse schema (player_id, position, age, team, season)."""
        from src.data.sleeper_client import SleeperClient

        client = SleeperClient()
        all_players = client.get_all_players()
        season_start = datetime.date(season, 9, 1)

        rows: list[dict] = []
        for player_id, info in all_players.items():
            pos = info.get("position")
            if pos not in self.FANTASY_POSITIONS:
                continue

            # Calculate age at the start of the season from birth_date
            age: int | None = None
            birth_date_str = info.get("birth_date")
            if birth_date_str:
                try:
                    bd = datetime.date.fromisoformat(birth_date_str)
                    age = (season_start - bd).days // 365
                except (ValueError, TypeError):
                    pass
            if age is None:
                age = info.get("age")

            rows.append({
                "player_id": str(player_id),
                "position": pos,
                "age": age,
                "team": info.get("team") or "FA",
                "season": season,
            })

        return pd.DataFrame(rows)

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
