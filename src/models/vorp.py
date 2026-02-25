"""
VORP (Value Over Replacement Player) Model.

Calculates fantasy value relative to a replacement-level player at
each position. Think of it as the opportunity cost framework from
economics — the replacement player is your next-best alternative,
like the risk-free rate in CAPM but for fantasy.

Dynasty VORP extends this with age-adjusted discounting, essentially
applying NPV (Net Present Value) to future fantasy production.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class LeagueSettings:
    """Your league's configuration for VORP calculation."""
    num_teams: int = 12
    qb_slots: int = 1
    rb_slots: int = 2
    wr_slots: int = 2
    te_slots: int = 1
    flex_slots: int = 2
    superflex_slots: int = 0
    bench_slots: int = 6
    scoring: str = "ppr"


# Positional aging decay factors by age bracket.
# retention = avg(PPG at age N+1) / avg(PPG at age N)
AGING_CURVES = {
    "QB": {
        (22, 26): 1.05,  # improving
        (27, 31): 1.00,  # peak/plateau
        (32, 35): 0.95,  # gradual decline
        (36, 45): 0.88,  # steep decline
    },
    "RB": {
        (20, 23): 1.08,  # rapid improvement
        (24, 26): 1.00,  # peak
        (27, 28): 0.90,  # the cliff starts
        (29, 35): 0.80,  # steep decline
    },
    "WR": {
        (20, 23): 1.06,
        (24, 28): 1.01,  # long peak
        (29, 31): 0.95,
        (32, 38): 0.88,
    },
    "TE": {
        (21, 24): 1.04,  # slow development
        (25, 29): 1.01,  # peak
        (30, 32): 0.95,
        (33, 38): 0.87,
    },
}


def get_retention_factor(position: str, age: int) -> float:
    """Get the year-over-year production retention factor for a position/age."""
    curves = AGING_CURVES.get(position, {})
    for (low, high), factor in curves.items():
        if low <= age <= high:
            return factor
    return 0.85  # default: decline


class VORPCalculator:
    """
    Calculates VORP and Dynasty VORP for player rankings.

    VORP = Player's PPG - Replacement Level PPG at their position

    Dynasty VORP = Sum of discounted future VORP over a projection window,
    where the discount comes from aging curves (like DCF discounting).
    """

    def __init__(self, settings: LeagueSettings | None = None):
        self.settings = settings or LeagueSettings()

    def _calculate_replacement_level(
        self, player_pool: pd.DataFrame
    ) -> dict[str, float]:
        """
        Determine the replacement-level PPG for each position.

        Replacement level = the PPG of the best player NOT rostered.
        In a 12-team league with 2 RB slots + 2 flex, that's roughly
        the RB37-RB40 range (accounting for some flex usage).
        """
        s = self.settings
        # Estimate how many starters per position across the league
        starters = {
            "QB": s.num_teams * (s.qb_slots + s.superflex_slots),
            "RB": s.num_teams * (s.rb_slots + round(s.flex_slots * 0.5)),
            "WR": s.num_teams * (s.wr_slots + round(s.flex_slots * 0.4)),
            "TE": s.num_teams * (s.te_slots + round(s.flex_slots * 0.1)),
        }

        replacement_levels = {}
        for pos, num_starters in starters.items():
            pos_players = (
                player_pool[player_pool["position"] == pos]
                .sort_values("ppg", ascending=False)
                .reset_index(drop=True)
            )
            # Replacement level = player just outside starter range
            repl_idx = min(num_starters, len(pos_players) - 1)
            if repl_idx >= 0 and len(pos_players) > repl_idx:
                replacement_levels[pos] = pos_players.iloc[repl_idx]["ppg"]
            else:
                replacement_levels[pos] = 0.0

            logger.debug(
                f"{pos}: {num_starters} starters, "
                f"replacement at #{repl_idx + 1} = {replacement_levels[pos]:.1f} PPG"
            )

        return replacement_levels

    def calculate_vorp(self, player_pool: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate single-season VORP for all players.

        Args:
            player_pool: DataFrame with columns [player_id, player_name,
                         position, ppg, total_points, games_played, age]

        Returns:
            DataFrame with added 'vorp' and 'replacement_ppg' columns.
        """
        replacement_levels = self._calculate_replacement_level(player_pool)

        df = player_pool.copy()
        df["replacement_ppg"] = df["position"].map(replacement_levels)
        df["vorp"] = df["ppg"] - df["replacement_ppg"]
        df["vorp_total"] = df["vorp"] * df["games_played"]

        return df.sort_values("vorp", ascending=False).reset_index(drop=True)

    def calculate_dynasty_vorp(
        self,
        player_pool: pd.DataFrame,
        projection_years: int = 5,
        discount_rate: float = 0.10,
    ) -> pd.DataFrame:
        """
        Calculate Dynasty VORP — the NPV of future fantasy production.

        This discounts future production by:
        1. Age-based decline curves (position-specific)
        2. A time-value discount rate (like a DCF model)

        Args:
            player_pool: DataFrame with [player_id, player_name, position,
                         ppg, total_points, games_played, age]
            projection_years: How many years to project forward
            discount_rate: Annual discount rate for time value (0.10 = 10%)

        Returns:
            DataFrame with dynasty_vorp column added.
        """
        # First get current VORP
        df = self.calculate_vorp(player_pool)

        dynasty_vorps = []
        for _, row in df.iterrows():
            current_vorp = row["vorp"]
            current_age = row["age"] if pd.notna(row["age"]) else 26
            position = row["position"]

            # Project future VORP using aging curves
            total_dynasty_vorp = current_vorp  # year 0 (current season)
            projected_vorp = current_vorp

            for year in range(1, projection_years + 1):
                future_age = current_age + year
                retention = get_retention_factor(position, future_age)
                projected_vorp *= retention

                # Discount for time value
                discounted = projected_vorp / ((1 + discount_rate) ** year)
                total_dynasty_vorp += discounted

            dynasty_vorps.append(total_dynasty_vorp)

        df["dynasty_vorp"] = dynasty_vorps
        df["dynasty_rank"] = df["dynasty_vorp"].rank(ascending=False).astype(int)

        return df.sort_values("dynasty_vorp", ascending=False).reset_index(drop=True)

    def positional_scarcity(self, player_pool: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate positional scarcity — how steep the talent drop-off
        is at each position. Higher scarcity = draft earlier.

        Uses a Gini-like concentration metric on VORP.
        """
        df = self.calculate_vorp(player_pool)

        results = []
        for pos in ["QB", "RB", "WR", "TE"]:
            pos_df = df[df["position"] == pos].head(36)
            if len(pos_df) < 5:
                continue

            vorps = pos_df["vorp"].values
            vorps = vorps[vorps > 0]

            if len(vorps) < 3:
                continue

            # Concentration: what % of total VORP is held by top players
            total_vorp = vorps.sum()
            top5_share = vorps[:5].sum() / total_vorp if total_vorp > 0 else 0
            top12_share = vorps[:12].sum() / total_vorp if total_vorp > 0 else 0

            # Drop-off: difference between #1 and median starter
            median_idx = min(12, len(vorps) - 1)
            dropoff = vorps[0] - vorps[median_idx]

            results.append({
                "position": pos,
                "top5_vorp_share": top5_share,
                "top12_vorp_share": top12_share,
                "max_vorp": vorps[0],
                "median_starter_vorp": vorps[median_idx],
                "dropoff": dropoff,
                "scarcity_score": top5_share * dropoff,
            })

        return pd.DataFrame(results).sort_values(
            "scarcity_score", ascending=False
        )
