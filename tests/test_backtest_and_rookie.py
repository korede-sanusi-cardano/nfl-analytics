"""
Tests for backtest simulator and rookie valuation model.

Run with: pytest tests/test_backtest_and_rookie.py -v

These tests use synthetic data so they run without nfl_data_py or network access.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.backtest import (
    DraftResult,
    DraftSettings,
    RedraftBacktester,
)
from src.models.rookie_model import (
    RookieValuationModel,
    build_awards_dataframe,
    get_award_counts,
    COLLEGE_AWARDS,
    COMBINE_BENCHMARKS,
)
from src.models.vorp import LeagueSettings, VORPCalculator


# ══════════════════════════════════════════════════════════════════════
# Fixtures — synthetic data that mirrors real nfl_data_py output
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def player_pool():
    """
    Synthetic player pool mimicking the output of NFLStats.get_top_performers().

    Columns match what VORPCalculator and RedraftBacktester expect:
    player_id, player_name, position, ppg, total_points, games_played, age
    """
    np.random.seed(42)
    players = []
    player_id = 1

    position_configs = [
        ("QB", 30, 18.0, 3.0),
        ("RB", 50, 14.0, 4.0),
        ("WR", 50, 12.0, 3.5),
        ("TE", 24, 9.0, 3.0),
    ]

    for pos, count, mean_ppg, std_ppg in position_configs:
        for i in range(count):
            ppg = max(1.0, np.random.normal(mean_ppg - i * 0.3, std_ppg))
            games = np.random.randint(10, 18)
            players.append(
                {
                    "player_id": f"P{player_id:04d}",
                    "player_name": f"{pos}{i + 1}",
                    "position": pos,
                    "ppg": round(ppg, 2),
                    "total_points": round(ppg * games, 1),
                    "games_played": games,
                    "age": np.random.randint(22, 35),
                }
            )
            player_id += 1

    return pd.DataFrame(players)


@pytest.fixture
def vorp_calculator():
    return VORPCalculator(LeagueSettings(num_teams=12))


@pytest.fixture
def draft_settings():
    return DraftSettings(num_teams=12, rounds=15)


@pytest.fixture
def awards_df():
    return build_awards_dataframe()


# ══════════════════════════════════════════════════════════════════════
# DraftSettings tests
# ══════════════════════════════════════════════════════════════════════


class TestDraftSettings:
    def test_defaults(self):
        ds = DraftSettings()
        assert ds.num_teams == 12
        assert ds.rounds == 15
        assert ds.scoring == "ppr"

    def test_roster_size_equals_rounds(self):
        ds = DraftSettings(rounds=16)
        assert ds.roster_size == 16

    def test_starter_slots(self):
        ds = DraftSettings()
        slots = ds.starter_slots
        assert slots["QB"] == 1
        assert slots["RB"] == 2
        assert slots["WR"] == 2
        assert slots["TE"] == 1

    def test_custom_settings(self):
        ds = DraftSettings(num_teams=10, rounds=18, qb_slots=2)
        assert ds.num_teams == 10
        assert ds.starter_slots["QB"] == 2


# ══════════════════════════════════════════════════════════════════════
# DraftResult tests
# ══════════════════════════════════════════════════════════════════════


class TestDraftResult:
    def test_repr(self):
        result = DraftResult(
            pick_position=7,
            roster=pd.DataFrame(),
            total_actual_points=1500.0,
            total_projected_points=1400.0,
            strategy="vorp",
        )
        text = repr(result)
        assert "pick=7" in text
        assert "vorp" in text
        assert "1500.0" in text

    def test_draft_log_default_empty(self):
        result = DraftResult(
            pick_position=1,
            roster=pd.DataFrame(),
            total_actual_points=0,
            total_projected_points=0,
            strategy="adp",
        )
        assert result.draft_log == []


# ══════════════════════════════════════════════════════════════════════
# Draft simulation tests (using a mock NFLStats)
# ══════════════════════════════════════════════════════════════════════


class _MockNFLStats:
    """Mock NFLStats that returns synthetic fixture data."""

    def __init__(self, pool):
        self._pool = pool

    def get_top_performers(self, season, position, top_n=60):
        df = self._pool[self._pool["position"] == position].head(top_n).copy()
        df["season"] = season
        return df

    def get_rosters(self, seasons):
        return pd.DataFrame()


class TestDraftSimulation:
    """Test the draft simulation engine in isolation."""

    def _make_backtester(self, player_pool, settings=None):
        settings = settings or DraftSettings()
        mock_stats = _MockNFLStats(player_pool)
        calc = VORPCalculator(LeagueSettings(num_teams=settings.num_teams))
        return RedraftBacktester(mock_stats, calc, settings)

    def _get_rankings(self, bt, rounds=None):
        rankings = bt._build_pre_season_rankings(target_season=2023, lookback_years=1)
        rankings["vorp_rank"] = range(1, len(rankings) + 1)
        return rankings

    def test_snake_draft_correct_pick_count(self, player_pool):
        settings = DraftSettings(num_teams=12, rounds=5)
        bt = self._make_backtester(player_pool, settings)
        rankings = self._get_rankings(bt)

        picks = bt._simulate_snake_draft(
            rankings, "vorp_rank", pick_position=1, strategy_name="vorp"
        )
        assert len(picks) == 5

    def test_snake_draft_no_duplicates(self, player_pool):
        settings = DraftSettings(num_teams=12, rounds=10)
        bt = self._make_backtester(player_pool, settings)
        rankings = self._get_rankings(bt)

        picks = bt._simulate_snake_draft(
            rankings, "vorp_rank", pick_position=6, strategy_name="vorp"
        )
        player_ids = [p["player_id"] for p in picks]
        assert len(player_ids) == len(set(player_ids)), "Duplicate players drafted"

    def test_all_pick_positions_produce_rosters(self, player_pool):
        settings = DraftSettings(num_teams=12, rounds=3)
        bt = self._make_backtester(player_pool, settings)
        rankings = self._get_rankings(bt)

        for pos in range(1, 13):
            picks = bt._simulate_snake_draft(
                rankings, "vorp_rank", pick_position=pos, strategy_name="vorp"
            )
            assert len(picks) == 3, f"Pick position {pos} produced {len(picks)} picks"

    def test_pick_position_1_gets_best_player(self, player_pool):
        settings = DraftSettings(num_teams=12, rounds=1)
        bt = self._make_backtester(player_pool, settings)
        rankings = self._get_rankings(bt)

        picks = bt._simulate_snake_draft(
            rankings, "vorp_rank", pick_position=1, strategy_name="vorp"
        )
        assert picks[0]["rank_value"] == 1


# ══════════════════════════════════════════════════════════════════════
# Lineup optimisation tests
# ══════════════════════════════════════════════════════════════════════


class TestLineupOptimisation:
    def _make_backtester_shell(self):
        bt = RedraftBacktester.__new__(RedraftBacktester)
        bt.draft_settings = DraftSettings()
        return bt

    def test_optimise_lineup_selects_best_starters(self):
        bt = self._make_backtester_shell()

        roster = pd.DataFrame(
            {
                "position": ["QB", "QB", "RB", "RB", "RB", "WR", "WR", "WR", "TE", "TE"],
                "total_points_actual": [300, 200, 250, 200, 150, 220, 180, 100, 160, 80],
            }
        )

        total = bt._optimise_lineup(roster)
        assert total > 0
        assert total >= 300  # best QB must be included

    def test_optimise_lineup_handles_thin_roster(self):
        bt = self._make_backtester_shell()

        roster = pd.DataFrame(
            {
                "position": ["QB", "RB", "WR", "TE"],
                "total_points_actual": [300, 250, 220, 160],
            }
        )

        total = bt._optimise_lineup(roster)
        assert total == 930


# ══════════════════════════════════════════════════════════════════════
# Roster scoring tests
# ══════════════════════════════════════════════════════════════════════


class TestScoringRoster:
    def _make_backtester_shell(self):
        bt = RedraftBacktester.__new__(RedraftBacktester)
        bt.draft_settings = DraftSettings()
        return bt

    def test_score_roster_maps_actual_points(self):
        bt = self._make_backtester_shell()

        draft_log = [
            {
                "player_id": "P0001", "player_name": "QB1", "position": "QB",
                "projected_ppg": 20, "projected_total": 340, "vorp": 10,
                "rank_value": 1, "round": 1, "overall_pick": 1, "team": 1,
            },
        ]
        actual_points = pd.DataFrame(
            {"player_id": ["P0001"], "total_points": [350.0], "ppg": [20.6], "games_played": [17]}
        )

        total, roster_df = bt._score_roster(draft_log, actual_points)
        assert roster_df["total_points_actual"].iloc[0] == 350.0

    def test_score_roster_handles_missing_players(self):
        bt = self._make_backtester_shell()

        draft_log = [
            {
                "player_id": "MISSING", "player_name": "Ghost", "position": "RB",
                "projected_ppg": 15, "projected_total": 255, "vorp": 8,
                "rank_value": 5, "round": 1, "overall_pick": 5, "team": 1,
            },
        ]
        actual_points = pd.DataFrame(
            {"player_id": ["OTHER"], "total_points": [200.0], "ppg": [12.0], "games_played": [17]}
        )

        total, roster_df = bt._score_roster(draft_log, actual_points)
        assert roster_df["total_points_actual"].iloc[0] == 0


# ══════════════════════════════════════════════════════════════════════
# Summary compilation tests
# ══════════════════════════════════════════════════════════════════════


class TestSummaryCompilation:
    def _make_results(self, strategy, points_list):
        return [
            DraftResult(
                pick_position=i + 1,
                roster=pd.DataFrame(),
                total_actual_points=pts,
                total_projected_points=1400,
                strategy=strategy,
            )
            for i, pts in enumerate(points_list)
        ]

    def test_compile_summary_structure(self):
        bt = RedraftBacktester.__new__(RedraftBacktester)
        vorp = self._make_results("vorp", [1510, 1520, 1530])
        adp = self._make_results("adp", [1455, 1460, 1465])

        summary = bt._compile_summary(vorp, adp, 2023)
        assert isinstance(summary, pd.DataFrame)
        assert "metric" in summary.columns
        assert "value" in summary.columns
        assert len(summary) == 10

    def test_pick_comparison_alpha(self):
        bt = RedraftBacktester.__new__(RedraftBacktester)
        vorp = self._make_results("vorp", [1600, 1400])
        adp = self._make_results("adp", [1500, 1450])

        comparison = bt._build_pick_comparison(vorp, adp)
        assert len(comparison) == 2
        assert comparison.iloc[0]["alpha"] == 100
        assert comparison.iloc[1]["alpha"] == -50
        assert comparison.iloc[0]["vorp_wins"] is True
        assert comparison.iloc[1]["vorp_wins"] is False


# ══════════════════════════════════════════════════════════════════════
# Rookie model — awards system
# ══════════════════════════════════════════════════════════════════════


class TestAwardsData:
    def test_awards_dataframe_not_empty(self, awards_df):
        assert len(awards_df) > 0

    def test_awards_have_required_columns(self, awards_df):
        required = {"college_season", "player_name", "award", "position"}
        assert required.issubset(set(awards_df.columns))

    def test_all_award_categories_present(self, awards_df):
        expected = {
            "Heisman", "Maxwell", "Walter Camp", "Biletnikoff",
            "Doak Walker", "Davey O'Brien", "John Mackey", "Johnny Unitas",
        }
        actual = set(awards_df["award"].unique())
        assert expected.issubset(actual)

    def test_heisman_winners_correct(self, awards_df):
        heisman = awards_df[awards_df["award"] == "Heisman"]
        winners = dict(zip(heisman["college_season"], heisman["player_name"]))
        assert winners[2019] == "Joe Burrow"
        assert winners[2022] == "Caleb Williams"
        assert winners[2023] == "Jayden Daniels"

    def test_position_awards_match_positions(self, awards_df):
        biletnikoff = awards_df[awards_df["award"] == "Biletnikoff"]
        assert (biletnikoff["position"] == "WR").all()

        doak_walker = awards_df[awards_df["award"] == "Doak Walker"]
        assert (doak_walker["position"] == "RB").all()

        mackey = awards_df[awards_df["award"] == "John Mackey"]
        assert (mackey["position"] == "TE").all()


class TestAwardCounting:
    def test_joe_burrow_awards(self, awards_df):
        counts = get_award_counts("Joe Burrow", 2019, awards_df)
        assert counts["total_awards"] >= 4
        assert counts["has_heisman"] == 1
        assert counts["has_position_award"] == 1
        assert counts["has_maxwell_or_camp"] == 1

    def test_unknown_player_zero(self, awards_df):
        counts = get_award_counts("Nobody McFakeName", 2023, awards_df)
        assert counts["total_awards"] == 0
        assert counts["has_heisman"] == 0

    def test_season_boundary_respected(self, awards_df):
        # Caleb Williams won 2022 Heisman — shouldn't count for 2021
        counts_before = get_award_counts("Caleb Williams", 2021, awards_df)
        assert counts_before["has_heisman"] == 0

        counts_after = get_award_counts("Caleb Williams", 2022, awards_df)
        assert counts_after["has_heisman"] == 1


# ══════════════════════════════════════════════════════════════════════
# Rookie model — combine benchmarks
# ══════════════════════════════════════════════════════════════════════


class TestCombineBenchmarks:
    def test_all_positions_have_benchmarks(self):
        for pos in ["QB", "RB", "WR", "TE"]:
            assert pos in COMBINE_BENCHMARKS

    def test_values_reasonable(self):
        for pos, bm in COMBINE_BENCHMARKS.items():
            assert 4.2 <= bm["forty"] <= 5.2, f"{pos} forty out of range"
            assert 25 <= bm["vertical"] <= 42, f"{pos} vertical out of range"
            assert 5 <= bm["bench"] <= 30, f"{pos} bench out of range"

    def test_speed_ordering(self):
        assert COMBINE_BENCHMARKS["RB"]["forty"] < COMBINE_BENCHMARKS["QB"]["forty"]
        assert COMBINE_BENCHMARKS["WR"]["forty"] < COMBINE_BENCHMARKS["TE"]["forty"]


# ══════════════════════════════════════════════════════════════════════
# Rookie model — feature extraction
# ══════════════════════════════════════════════════════════════════════


class TestRookieFeatureExtraction:
    def _make_prospect(self):
        return pd.DataFrame([{
            "player_name": "Joe Burrow", "position": "QB",
            "draft_round": 1, "draft_pick": 1,
            "forty": 4.89, "vertical": 32.0, "bench": 19,
            "broad_jump": 108, "three_cone": 7.12,
            "total_awards": 5, "has_heisman": 1,
            "has_position_award": 1, "has_maxwell_or_camp": 1,
            "draft_year": 2020, "age": 23, "ppg": 0,
            "total_points": 0, "games_played": 0, "player_id": "BURROW",
        }])

    def test_correct_columns(self):
        model = RookieValuationModel.__new__(RookieValuationModel)
        model._feature_columns = []
        features = model._extract_features(self._make_prospect())

        assert "draft_round" in features.columns
        assert "is_first_round" in features.columns
        assert "combine_forty" in features.columns
        assert "pos_QB" in features.columns
        assert "total_awards" in features.columns
        assert "draft_pick_x_awards" in features.columns

    def test_first_round_flag(self):
        model = RookieValuationModel.__new__(RookieValuationModel)
        model._feature_columns = []
        features = model._extract_features(self._make_prospect())
        assert features["is_first_round"].iloc[0] == 1
        assert features["is_top_10"].iloc[0] == 1

    def test_position_encoding(self):
        model = RookieValuationModel.__new__(RookieValuationModel)
        model._feature_columns = []
        features = model._extract_features(self._make_prospect())
        assert features["pos_QB"].iloc[0] == 1
        assert features["pos_RB"].iloc[0] == 0

    def test_no_nans(self):
        model = RookieValuationModel.__new__(RookieValuationModel)
        model._feature_columns = []
        features = model._extract_features(self._make_prospect())
        assert features.isna().sum().sum() == 0


# ══════════════════════════════════════════════════════════════════════
# Rookie model — dynasty VORP integration
# ══════════════════════════════════════════════════════════════════════


class TestDynastyVORPIntegration:
    def _make_model(self):
        return RookieValuationModel.__new__(RookieValuationModel)

    def _make_veterans(self):
        return pd.DataFrame({
            "player_name": ["Vet A", "Vet B"],
            "position": ["RB", "WR"],
            "ppg": [18.0, 16.0],
            "total_points": [306, 272],
            "games_played": [17, 17],
            "age": [27, 25],
        })

    def _make_rookies(self, **overrides):
        data = {
            "player_name": ["Rookie WR1"],
            "position": ["WR"],
            "projected_yr1_ppg": [12.0],
            "projected_yr1_total": [204],
            "confidence": [0.8],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_merge_adds_rookies(self):
        model = self._make_model()
        combined = model.merge_with_dynasty_vorp(self._make_rookies(), self._make_veterans())
        assert len(combined) == 3
        assert "Rookie WR1" in combined["player_name"].values

    def test_confidence_discount(self):
        model = self._make_model()

        high = self._make_rookies(
            player_name=["High"], projected_yr1_ppg=[15.0], confidence=[1.0]
        )
        low = self._make_rookies(
            player_name=["Low"], projected_yr1_ppg=[15.0], confidence=[0.0]
        )

        high_combined = model.merge_with_dynasty_vorp(high, self._make_veterans())
        low_combined = model.merge_with_dynasty_vorp(low, self._make_veterans())

        high_ppg = high_combined[high_combined["player_name"] == "High"]["ppg"].iloc[0]
        low_ppg = low_combined[low_combined["player_name"] == "Low"]["ppg"].iloc[0]

        # High: 15.0 * (0.7 + 0.3 * 1.0) = 15.0
        # Low:  15.0 * (0.7 + 0.3 * 0.0) = 10.5
        assert high_ppg == pytest.approx(15.0, abs=0.1)
        assert low_ppg == pytest.approx(10.5, abs=0.1)

    def test_empty_rookies_returns_veterans(self):
        model = self._make_model()
        combined = model.merge_with_dynasty_vorp(pd.DataFrame(), self._make_veterans())
        assert len(combined) == 2


# ══════════════════════════════════════════════════════════════════════
# Confidence calculation tests
# ══════════════════════════════════════════════════════════════════════


class TestConfidenceCalculation:
    def test_full_data_high_confidence(self):
        model = RookieValuationModel.__new__(RookieValuationModel)
        features = pd.DataFrame({
            "draft_pick": [1.0], "draft_round": [1.0],
            "combine_forty": [4.5], "combine_vertical": [35.0],
            "combine_bench": [20.0], "combine_broad_jump": [120.0],
            "combine_three_cone": [7.0],
        })
        confidence = model._calculate_confidence(features)
        assert confidence.iloc[0] >= 0.8

    def test_missing_data_lower_confidence(self):
        model = RookieValuationModel.__new__(RookieValuationModel)
        features = pd.DataFrame({
            "draft_pick": [0.0], "draft_round": [np.nan],
            "combine_forty": [np.nan], "combine_vertical": [np.nan],
            "combine_bench": [np.nan], "combine_broad_jump": [np.nan],
            "combine_three_cone": [np.nan],
        })
        confidence = model._calculate_confidence(features)
        assert confidence.iloc[0] < 0.5
