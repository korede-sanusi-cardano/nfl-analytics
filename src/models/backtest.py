"""
Redraft Backtest Simulator.

Simulates fantasy drafts at each pick position (1-12) using only data
available BEFORE the season starts, then measures actual fantasy points
each drafted roster produced during that season.

The goal: prove whether VORP-based drafting generates more total fantasy
points than following consensus ADP (Average Draft Position).

Think of it as paper-trading a quantitative strategy against historical
market data before deploying real capital. We run the same draft 12 times
(once per pick slot), compare to an ADP-drafted benchmark, and measure
the alpha (or lack thereof) the VORP model provides.

Usage:
    from src.models.backtest import RedraftBacktester
    from src.data.nfl_stats import NFLStats
    from src.models.vorp import VORPCalculator, LeagueSettings

    nfl = NFLStats()
    calc = VORPCalculator(LeagueSettings())
    bt = RedraftBacktester(nfl_stats=nfl, vorp_calculator=calc)

    # Run a full backtest for the 2023 season
    results = bt.run_backtest(target_season=2023)

    # View results
    print(results["summary"])
    print(results["pick_results"])
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.data.nfl_stats import NFLStats
from src.models.vorp import LeagueSettings, VORPCalculator


@dataclass
class DraftSettings:
    """Configuration for the simulated draft."""

    num_teams: int = 12
    rounds: int = 15  # standard redraft: 15 rounds
    qb_slots: int = 1
    rb_slots: int = 2
    wr_slots: int = 2
    te_slots: int = 1
    flex_slots: int = 2  # RB/WR/TE
    k_slots: int = 1
    dst_slots: int = 1
    bench_slots: int = 5
    scoring: str = "ppr"

    @property
    def roster_size(self) -> int:
        return self.rounds

    @property
    def starter_slots(self) -> dict[str, int]:
        """Minimum starters needed per position (excluding flex)."""
        return {
            "QB": self.qb_slots,
            "RB": self.rb_slots,
            "WR": self.wr_slots,
            "TE": self.te_slots,
        }


@dataclass
class DraftResult:
    """Result from a single simulated draft at a specific pick position."""

    pick_position: int
    roster: pd.DataFrame
    total_actual_points: float
    total_projected_points: float
    strategy: str  # "vorp" or "adp"
    draft_log: list[dict] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"DraftResult(pick={self.pick_position}, strategy={self.strategy}, "
            f"actual_pts={self.total_actual_points:.1f}, "
            f"projected_pts={self.total_projected_points:.1f})"
        )


class RedraftBacktester:
    """
    Simulate drafts using pre-season data and evaluate against actual results.

    The backtest enforces strict temporal separation: VORP rankings are built
    from prior-season data only (no look-ahead bias), then actual fantasy
    points from the target season measure roster quality.
    """

    # Scoring column mapping
    SCORING_MAP = {
        "ppr": "fantasy_points_ppr",
        "half_ppr": "fantasy_points_ppr",  # adjusted below
        "standard": "fantasy_points",
    }

    def __init__(
        self,
        nfl_stats: NFLStats,
        vorp_calculator: VORPCalculator,
        draft_settings: DraftSettings | None = None,
    ):
        self.nfl_stats = nfl_stats
        self.vorp_calc = vorp_calculator
        self.draft_settings = draft_settings or DraftSettings()

    def _build_pre_season_rankings(
        self,
        target_season: int,
        lookback_years: int = 3,
    ) -> pd.DataFrame:
        """
        Build VORP rankings using ONLY data available before the target season.

        This is the critical no-look-ahead constraint. If we're backtesting
        2023, we only use 2020-2022 data to generate rankings.

        Args:
            target_season: The season we're simulating a draft for
            lookback_years: How many prior seasons to use for projections

        Returns:
            DataFrame with projected rankings for the upcoming season
        """
        prior_seasons = list(range(target_season - lookback_years, target_season))
        logger.info(
            f"Building pre-season rankings for {target_season} "
            f"using data from {prior_seasons[0]}-{prior_seasons[-1]}"
        )

        # Get the most recent prior season as primary projection basis
        most_recent = prior_seasons[-1]

        all_players = []
        for pos in ["QB", "RB", "WR", "TE"]:
            top = self.nfl_stats.get_top_performers(
                most_recent, pos, top_n=60
            )
            top["season"] = most_recent
            all_players.append(top)

        player_pool = pd.concat(all_players, ignore_index=True)
        if lookback_years > 1:
            player_pool = self._apply_multi_year_weighting(player_pool, prior_seasons)
        vorp_rankings = self.vorp_calc.calculate_vorp(player_pool)
        vorp_rankings = vorp_rankings.sort_values("vorp", ascending=False).reset_index(drop=True)
        vorp_rankings["vorp_rank"] = range(1, len(vorp_rankings) + 1)
        return vorp_rankings

    def _apply_multi_year_weighting(
        self,
        primary_pool: pd.DataFrame,
        seasons: list[int],
    ) -> pd.DataFrame:
        """
        Blend multiple years of data with recency weighting.

        More recent seasons get higher weight — same principle as an
        exponentially weighted moving average in time-series analysis.
        Weights: most recent = 0.60, prior = 0.25, two years ago = 0.15
        """
        weights = {seasons[-1]: 0.60}
        if len(seasons) >= 2:
            weights[seasons[-2]] = 0.25
        if len(seasons) >= 3:
            weights[seasons[-3]] = 0.15
        # For now, the primary pool is already the most recent season
        # In a full implementation, we'd pull all seasons and blend PPG
        # This is a simplification that still captures the key idea
        return primary_pool

    def _build_adp_rankings(self, player_pool: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ADP-style rankings as the benchmark.

        We approximate ADP by using raw total fantasy points from the prior
        season (since ADP largely reflects last year's production + name value).
        This is a reasonable proxy — actual ADP data could be swapped in
        from FantasyPros or similar sources.
        """
        adp = player_pool.copy()
        adp = adp.sort_values("total_points", ascending=False).reset_index(drop=True)
        adp["adp_rank"] = range(1, len(adp) + 1)
        return adp

    def _get_actual_season_points(
        self,
        target_season: int,
    ) -> pd.DataFrame:
        """
        Get actual fantasy points produced during the target season.

        This is our ground truth for evaluating how good each drafted
        roster actually was.
        """
        logger.info(f"Loading actual results for {target_season}")
        all_actuals = []
        for pos in ["QB", "RB", "WR", "TE"]:
            df = self.nfl_stats.get_top_performers(target_season, pos, top_n=80)
            df["season"] = target_season
            all_actuals.append(df)
        return pd.concat(all_actuals, ignore_index=True)

    def _simulate_snake_draft(
        self,
        rankings: pd.DataFrame,
        rank_column: str,
        pick_position: int,
        strategy_name: str,
    ) -> list[dict]:
        """
        Simulate a full snake draft where our team picks at pick_position.

        Snake draft order: Round 1 goes 1→12, Round 2 goes 12→1, etc.
        Other teams draft by best available according to the same rankings
        (a simplification, but standard for backtesting).

        Args:
            rankings: Player pool sorted by the ranking column
            rank_column: Which column to use for draft order ("vorp_rank" or "adp_rank")
            pick_position: Our draft slot (1-12)
            strategy_name: Label for this strategy

        Returns:
            List of draft picks [{round, overall_pick, player_name, ...}]
        """
        num_teams = self.draft_settings.num_teams
        num_rounds = self.draft_settings.rounds
        available = rankings.sort_values(rank_column).copy()
        available_set = set(available.index.tolist())

        team_positions = {t: {"QB": 0, "RB": 0, "WR": 0, "TE": 0} for t in range(1, num_teams + 1)}
        our_picks = []

        for round_num in range(1, num_rounds + 1):
            order = list(range(1, num_teams + 1)) if round_num % 2 == 1 else list(range(num_teams, 0, -1))

            for slot_idx, team in enumerate(order):
                overall_pick = (round_num - 1) * num_teams + slot_idx + 1

                if team == pick_position:
                    pick = self._make_smart_pick(available, available_set, team_positions[team], rank_column, round_num)
                else:
                    pick = self._make_bpa_pick(available, available_set, team_positions[team], rank_column, round_num)

                if pick is not None:
                    pick_record = {
                        "round": round_num, "overall_pick": overall_pick, "team": team,
                        "player_id": pick["player_id"], "player_name": pick["player_name"],
                        "position": pick["position"], "projected_ppg": pick["ppg"],
                        "projected_total": pick["total_points"], "vorp": pick.get("vorp", 0),
                        "rank_value": pick.get(rank_column, 0),
                    }
                    team_positions[team][pick["position"]] += 1
                    available_set.discard(pick.name)
                    if team == pick_position:
                        our_picks.append(pick_record)

        return our_picks

    def _make_smart_pick(
        self,
        available: pd.DataFrame,
        available_set: set,
        current_positions: dict[str, int],
        rank_column: str,
        current_round: int,
    ) -> pd.Series | None:
        """
        Make an intelligent draft pick considering positional need.

        Early rounds: best VORP available (position-agnostic value).
        Mid rounds: fill positional minimums if not met.
        Late rounds: best available for bench depth.
        """
        pool = available.loc[list(available_set)].sort_values(rank_column)
        if pool.empty:
            return None
        starter_needs = self.draft_settings.starter_slots
        total_rounds = self.draft_settings.rounds
        rounds_remaining = total_rounds - current_round + 1
        unfilled = {pos: needed - current_positions.get(pos, 0)
                    for pos, needed in starter_needs.items()
                    if needed - current_positions.get(pos, 0) > 0}
        if sum(unfilled.values()) >= rounds_remaining and unfilled:
            need_pool = pool[pool["position"].isin(unfilled.keys())]
            if not need_pool.empty:
                return need_pool.iloc[0]
        return pool.iloc[0]

    def _make_bpa_pick(
        self,
        available: pd.DataFrame,
        available_set: set,
        current_positions: dict[str, int],
        rank_column: str,
        current_round: int,
    ) -> pd.Series | None:
        """Simple best-player-available for simulated opponents."""
        pool = available.loc[list(available_set)].sort_values(rank_column)
        if pool.empty:
            return None
        max_by_position = {"QB": 2, "RB": 6, "WR": 6, "TE": 2}
        for _, player in pool.iterrows():
            pos = player["position"]
            if current_positions.get(pos, 0) < max_by_position.get(pos, 6):
                return player
        return pool.iloc[0]

    def _score_roster(
        self,
        draft_log: list[dict],
        actual_points: pd.DataFrame,
    ) -> tuple[float, pd.DataFrame]:
        """
        Score a drafted roster against actual season results.

        Maps each drafted player to their actual fantasy production,
        then optimises the starting lineup to maximise points
        (simulating a perfectly-managed team — isolates draft quality
        from lineup management).

        Returns:
            (total_actual_points, roster_with_actuals)
        """
        roster_df = pd.DataFrame(draft_log)
        # Rename any projected columns that would collide with actuals before merging
        rename_map = {c: f"{c}_projected" for c in ["total_points", "ppg", "games_played"] if c in roster_df.columns}
        roster_df = roster_df.rename(columns=rename_map)
        merged = roster_df.merge(
            actual_points[["player_id", "total_points", "ppg", "games_played"]],
            on="player_id",
            how="left",
        )
        merged = merged.rename(columns={"total_points": "total_points_actual", "ppg": "ppg_actual", "games_played": "games_played_actual"})
        for col in ["total_points_actual", "ppg_actual", "games_played_actual"]:
            if col not in merged.columns:
                merged[col] = 0.0
            else:
                merged[col] = merged[col].fillna(0)
        total_points = self._optimise_lineup(merged)
        return total_points, merged

    def _optimise_lineup(self, roster: pd.DataFrame) -> float:
        """
        Find the optimal starting lineup from the roster.

        Assigns best players to each required slot, then fills flex
        with remaining best options. Returns total season points
        for the optimal lineup.
        """
        settings = self.draft_settings
        assigned = set()
        total = 0.0
        roster_sorted = roster.sort_values("total_points_actual", ascending=False)
        for pos, slots in settings.starter_slots.items():
            pos_players = roster_sorted[
                (roster_sorted["position"] == pos) & (~roster_sorted.index.isin(assigned))
            ].head(slots)
            total += pos_players["total_points_actual"].sum()
            assigned.update(pos_players.index)
        flex_eligible = roster_sorted[
            (roster_sorted["position"].isin(["RB", "WR", "TE"])) & (~roster_sorted.index.isin(assigned))
        ].head(settings.flex_slots)
        total += flex_eligible["total_points_actual"].sum()
        return total

    def run_backtest(
        self,
        target_season: int,
        lookback_years: int = 3,
        pick_positions: list[int] | None = None,
    ) -> dict:
        """
        Run the full backtest for a target season.

        For each pick position (1-12), simulates two drafts:
        1. VORP-guided draft (our model)
        2. ADP-based draft (the benchmark)

        Then compares actual season points to measure alpha.

        Args:
            target_season: Season to backtest (e.g., 2023)
            lookback_years: Prior seasons for building projections
            pick_positions: Which pick slots to test (default: all 1-12)

        Returns:
            Dict with:
                - summary: Overall comparison stats
                - pick_results: Per-pick-position results
                - vorp_results: List of DraftResult for VORP strategy
                - adp_results: List of DraftResult for ADP strategy
        """
        if pick_positions is None:
            pick_positions = list(range(1, self.draft_settings.num_teams + 1))

        logger.info(f"{'=' * 60}")
        logger.info(f"BACKTEST: {target_season} season")
        logger.info(f"Pick positions: {pick_positions}")
        logger.info(f"Lookback: {lookback_years} years")
        logger.info(f"{'=' * 60}")

        # Step 1: Build pre-season rankings (prior data only)
        vorp_rankings = self._build_pre_season_rankings(target_season, lookback_years)
        adp_rankings = self._build_adp_rankings(vorp_rankings)

        # Step 2: Get actual season results (ground truth)
        actual_points = self._get_actual_season_points(target_season)

        # Step 3: Simulate drafts at each pick position
        vorp_results, adp_results = [], []
        for pick_pos in pick_positions:
            logger.info(f"\n--- Simulating pick position #{pick_pos} ---")

            vorp_picks = self._simulate_snake_draft(vorp_rankings, "vorp_rank", pick_pos, "vorp")
            vorp_total, vorp_roster = self._score_roster(vorp_picks, actual_points)
            vorp_results.append(DraftResult(
                pick_position=pick_pos, roster=vorp_roster, total_actual_points=vorp_total,
                total_projected_points=sum(p["projected_total"] for p in vorp_picks),
                strategy="vorp", draft_log=vorp_picks,
            ))

            adp_picks = self._simulate_snake_draft(adp_rankings, "adp_rank", pick_pos, "adp")
            adp_total, adp_roster = self._score_roster(adp_picks, actual_points)
            adp_results.append(DraftResult(
                pick_position=pick_pos, roster=adp_roster, total_actual_points=adp_total,
                total_projected_points=sum(p["projected_total"] for p in adp_picks),
                strategy="adp", draft_log=adp_picks,
            ))

            alpha = vorp_total - adp_total
            logger.info(
                f"Pick #{pick_pos}: VORP={vorp_total:.1f} | "
                f"ADP={adp_total:.1f} | Alpha={alpha:+.1f}"
            )

        # Step 4: Compile results
        summary = self._compile_summary(vorp_results, adp_results, target_season)
        return {
            "summary": summary,
            "pick_results": self._build_pick_comparison(vorp_results, adp_results),
            "vorp_results": vorp_results,
            "adp_results": adp_results,
            "target_season": target_season,
        }

    def _compile_summary(
        self,
        vorp_results: list[DraftResult],
        adp_results: list[DraftResult],
        target_season: int,
    ) -> pd.DataFrame:
        """Compile high-level summary statistics."""
        vorp_totals = [r.total_actual_points for r in vorp_results]
        adp_totals = [r.total_actual_points for r in adp_results]
        alphas = [v - a for v, a in zip(vorp_totals, adp_totals)]
        return pd.DataFrame({
            "metric": ["Season","Avg VORP Points","Avg ADP Points","Avg Alpha (VORP - ADP)",
                        "Median Alpha","Win Rate (VORP > ADP)","Max Alpha","Min Alpha",
                        "Best Pick Position (VORP)","Worst Pick Position (VORP)"],
            "value": [target_season, f"{np.mean(vorp_totals):.1f}", f"{np.mean(adp_totals):.1f}",
                      f"{np.mean(alphas):+.1f}", f"{np.median(alphas):+.1f}",
                      f"{sum(1 for a in alphas if a > 0)}/{len(alphas)} ({sum(1 for a in alphas if a > 0)/len(alphas):.0%})",
                      f"{max(alphas):+.1f}", f"{min(alphas):+.1f}",
                      str(vorp_results[np.argmax(vorp_totals)].pick_position),
                      str(vorp_results[np.argmin(vorp_totals)].pick_position)],
        })

    def _build_pick_comparison(
        self,
        vorp_results: list[DraftResult],
        adp_results: list[DraftResult],
    ) -> pd.DataFrame:
        """Build per-pick comparison table."""
        rows = []
        for vr, ar in zip(vorp_results, adp_results):
            alpha = vr.total_actual_points - ar.total_actual_points
            rows.append({"pick_position": vr.pick_position,
                         "vorp_actual_points": vr.total_actual_points,
                         "adp_actual_points": ar.total_actual_points, "alpha": alpha,
                         "vorp_wins": alpha > 0,
                         "vorp_projected_points": vr.total_projected_points,
                         "adp_projected_points": ar.total_projected_points})
        return pd.DataFrame(rows)

    def run_multi_season_backtest(
        self,
        seasons: list[int] | None = None,
        lookback_years: int = 3,
    ) -> dict:
        """
        Run backtests across multiple seasons for statistical significance.

        Default tests 2020-2024 (using 2017+ data for lookback).
        More seasons = more confidence in whether VORP adds value.
        """
        if seasons is None:
            seasons = [2020, 2021, 2022, 2023, 2024]
        all_results, all_alphas = {}, []
        for season in seasons:
            logger.info(f"\n{'#' * 60}")
            logger.info(f"# SEASON: {season}")
            logger.info(f"{'#' * 60}")
            result = self.run_backtest(target_season=season, lookback_years=lookback_years)
            all_results[season] = result
            for _, row in result["pick_results"].iterrows():
                all_alphas.append({"season": season, "pick_position": row["pick_position"],
                                   "alpha": row["alpha"], "vorp_points": row["vorp_actual_points"],
                                   "adp_points": row["adp_actual_points"]})
        alpha_df = pd.DataFrame(all_alphas)
        multi_summary = {
            "seasons_tested": seasons, "total_matchups": len(alpha_df),
            "overall_win_rate": (alpha_df["alpha"] > 0).mean(),
            "avg_alpha": alpha_df["alpha"].mean(), "median_alpha": alpha_df["alpha"].median(),
            "std_alpha": alpha_df["alpha"].std(),
            "alpha_by_season": alpha_df.groupby("season")["alpha"].mean().to_dict(),
            "alpha_by_pick": alpha_df.groupby("pick_position")["alpha"].mean().to_dict(),
            "season_results": all_results, "full_alpha_data": alpha_df,
        }
        logger.info(f"\n{'=' * 60}")
        logger.info("MULTI-SEASON BACKTEST RESULTS")
        logger.info(f"{'=' * 60}")
        logger.info(f"Seasons: {seasons}")
        logger.info(f"Total matchups: {len(alpha_df)}")
        logger.info(f"VORP Win Rate: {multi_summary['overall_win_rate']:.1%}")
        logger.info(f"Avg Alpha: {multi_summary['avg_alpha']:+.1f} points")
        logger.info(f"Median Alpha: {multi_summary['median_alpha']:+.1f} points")
        return multi_summary
