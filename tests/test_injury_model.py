"""
Tests for the injury adjustment model.

Run with: pytest tests/test_injury_model.py -v
"""

import pandas as pd
import pytest

from src.models.injury_model import (
    INJURY_CATALOG,
    SLEEPER_STATUS_MAP,
    InjuryAdjuster,
    InjuryRecord,
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def adjuster():
    return InjuryAdjuster()


@pytest.fixture
def dynasty_df():
    """Minimal Dynasty VORP DataFrame for testing adjustments."""
    return pd.DataFrame([
        {
            "player_name": "Tyreek Hill", "position": "WR", "age": 31,
            "ppg": 16.0, "total_points": 272, "games_played": 17,
            "vorp": 8.5, "dynasty_vorp": 30.0,
        },
        {
            "player_name": "Patrick Mahomes", "position": "QB", "age": 29,
            "ppg": 22.0, "total_points": 374, "games_played": 17,
            "vorp": 12.0, "dynasty_vorp": 52.0,
        },
        {
            "player_name": "Breece Hall", "position": "RB", "age": 23,
            "ppg": 17.0, "total_points": 289, "games_played": 17,
            "vorp": 10.0, "dynasty_vorp": 45.0,
        },
    ])


# ══════════════════════════════════════════════════════════════════════
# Injury catalog tests
# ══════════════════════════════════════════════════════════════════════


class TestInjuryCatalog:
    def test_all_entries_have_required_fields(self):
        required = {"label", "games_missed", "retention_multiplier",
                     "recovery_seasons", "positional_modifier"}
        for key, entry in INJURY_CATALOG.items():
            assert required.issubset(set(entry.keys())), f"{key} missing fields"

    def test_severity_levels_consistent(self):
        severities = {"minor", "moderate", "major", "career_threatening"}
        for key, entry in INJURY_CATALOG.items():
            assert set(entry["games_missed"].keys()) == severities, f"{key} games_missed"
            assert set(entry["retention_multiplier"].keys()) == severities, f"{key} retention"

    def test_retention_multipliers_in_range(self):
        for key, entry in INJURY_CATALOG.items():
            for sev, mult in entry["retention_multiplier"].items():
                assert 0 < mult <= 1.0, f"{key}/{sev} retention out of range: {mult}"

    def test_games_missed_in_range(self):
        for key, entry in INJURY_CATALOG.items():
            for sev, games in entry["games_missed"].items():
                assert 0 <= games <= 17, f"{key}/{sev} games out of range: {games}"

    def test_worse_severity_means_more_impact(self):
        """More severe injuries should miss more games and have lower retention."""
        severity_order = ["minor", "moderate", "major", "career_threatening"]
        for key, entry in INJURY_CATALOG.items():
            games = [entry["games_missed"][s] for s in severity_order]
            retention = [entry["retention_multiplier"][s] for s in severity_order]
            # Games missed should be non-decreasing
            assert all(a <= b for a, b in zip(games, games[1:])), f"{key} games not increasing"
            # Retention should be non-increasing
            assert all(a >= b for a, b in zip(retention, retention[1:])), f"{key} retention not decreasing"


class TestSleeperStatusMap:
    def test_ir_maps_to_major(self):
        injury_type, severity = SLEEPER_STATUS_MAP["IR"]
        assert injury_type == "general_ir"
        assert severity == "major"

    def test_questionable_ignored(self):
        injury_type, severity = SLEEPER_STATUS_MAP["Questionable"]
        assert injury_type is None


# ══════════════════════════════════════════════════════════════════════
# Injury record tests
# ══════════════════════════════════════════════════════════════════════


class TestInjuryRecord:
    def test_months_since_injury(self):
        record = InjuryRecord(
            player_name="Test", position="WR",
            injury_type="acl", severity="major",
            injury_date="2024-01-01",
        )
        months = record.months_since_injury
        assert months is not None
        assert months > 12  # It's 2026, so > 12 months

    def test_no_date_returns_none(self):
        record = InjuryRecord(
            player_name="Test", position="WR",
            injury_type="acl", severity="major",
        )
        assert record.months_since_injury is None


# ══════════════════════════════════════════════════════════════════════
# Adjuster core tests
# ══════════════════════════════════════════════════════════════════════


class TestInjuryAdjuster:
    def test_add_injury(self, adjuster):
        adjuster.add_injury("Tyreek Hill", "WR", "broken_leg", "major")
        assert adjuster.injury_count == 1

    def test_add_invalid_injury_type(self, adjuster):
        adjuster.add_injury("Test", "WR", "fake_injury", "major")
        assert adjuster.injury_count == 0  # should be rejected

    def test_games_projection_healthy(self, adjuster):
        games = adjuster.get_games_projection("Healthy Player", "WR")
        assert games == 17

    def test_games_projection_broken_leg_major(self, adjuster):
        adjuster.add_injury(
            "Tyreek Hill", "WR", "broken_leg", "major",
            injury_date="2024-11-24", surgery=True,
        )
        games = adjuster.get_games_projection("Tyreek Hill", "WR")
        assert games < 17
        # Major broken leg with surgery — should project significant time missed
        # But injury is > 12 months ago by now (March 2026), so recovery may be done
        assert games >= 0

    def test_retention_adjustment_healthy(self, adjuster):
        adj = adjuster.get_retention_adjustment("Healthy Player", "WR", 28)
        assert adj == 1.0

    def test_retention_adjustment_injured(self, adjuster):
        adjuster.add_injury("Tyreek Hill", "WR", "broken_leg", "major")
        adj = adjuster.get_retention_adjustment("Tyreek Hill", "WR", 31, projection_year=1)
        assert adj < 1.0
        assert adj > 0.2

    def test_retention_past_recovery_window(self, adjuster):
        adjuster.add_injury("Test", "WR", "hamstring", "minor")
        # Hamstring minor has 1 recovery season
        adj_yr1 = adjuster.get_retention_adjustment("Test", "WR", 26, projection_year=1)
        adj_yr2 = adjuster.get_retention_adjustment("Test", "WR", 26, projection_year=2)
        assert adj_yr1 < 1.0
        assert adj_yr2 == 1.0  # past recovery window

    def test_age_penalty_applied(self, adjuster):
        adjuster.add_injury("Old WR", "WR", "acl", "major")
        adjuster.add_injury("Young WR", "WR", "acl", "major")
        old_adj = adjuster.get_retention_adjustment("Old WR", "WR", 33, 1)
        young_adj = adjuster.get_retention_adjustment("Young WR", "WR", 24, 1)
        assert old_adj < young_adj  # older player hit harder

    def test_surgery_penalty(self, adjuster):
        adjuster.add_injury("No Surgery", "WR", "acl", "major", surgery=False)
        adjuster.add_injury("Surgery", "WR", "acl", "major", surgery=True)
        no_surg = adjuster.get_retention_adjustment("No Surgery", "WR", 27, 1)
        surg = adjuster.get_retention_adjustment("Surgery", "WR", 27, 1)
        assert surg < no_surg


# ══════════════════════════════════════════════════════════════════════
# Dynasty VORP adjustment tests
# ══════════════════════════════════════════════════════════════════════


class TestDynastyVORPAdjustment:
    def test_healthy_player_unchanged(self, adjuster, dynasty_df):
        result = adjuster.adjust_dynasty_vorp(dynasty_df)
        mahomes = result[result["player_name"] == "Patrick Mahomes"].iloc[0]
        assert mahomes["injury_status"] == "Healthy"
        assert mahomes["injury_adjusted_vorp"] == mahomes["dynasty_vorp"]
        assert mahomes["injury_discount_pct"] == 0.0

    def test_injured_player_discounted(self, adjuster, dynasty_df):
        adjuster.add_injury(
            "Tyreek Hill", "WR", "broken_leg", "major",
            injury_date="2024-11-24", surgery=True,
        )
        result = adjuster.adjust_dynasty_vorp(dynasty_df)
        hill = result[result["player_name"] == "Tyreek Hill"].iloc[0]
        assert hill["injury_status"] != "Healthy"
        assert hill["injury_adjusted_vorp"] < hill["dynasty_vorp"]
        assert hill["injury_discount_pct"] > 0

    def test_result_has_required_columns(self, adjuster, dynasty_df):
        result = adjuster.adjust_dynasty_vorp(dynasty_df)
        required = {"injury_adjusted_vorp", "injury_status",
                     "games_projected", "injury_discount_pct", "injury_adj_rank"}
        assert required.issubset(set(result.columns))

    def test_rankings_reordered(self, adjuster, dynasty_df):
        adjuster.add_injury("Tyreek Hill", "WR", "broken_leg", "career_threatening")
        result = adjuster.adjust_dynasty_vorp(dynasty_df)
        # Hill should drop in rankings with career-threatening injury
        hill_rank = result[result["player_name"] == "Tyreek Hill"]["injury_adj_rank"].iloc[0]
        assert hill_rank > 1  # no longer top

    def test_manual_overrides_batch(self, adjuster):
        overrides = [
            {"player_name": "Player A", "position": "RB",
             "injury_type": "acl", "severity": "major"},
            {"player_name": "Player B", "position": "WR",
             "injury_type": "hamstring", "severity": "minor"},
        ]
        count = adjuster.load_manual_overrides(overrides)
        assert count == 2
        assert adjuster.injury_count == 2


# ══════════════════════════════════════════════════════════════════════
# Injury report tests
# ══════════════════════════════════════════════════════════════════════


class TestInjuryReport:
    def test_empty_report(self, adjuster):
        report = adjuster.injury_report()
        assert report.empty

    def test_report_contents(self, adjuster):
        adjuster.add_injury("Tyreek Hill", "WR", "broken_leg", "major",
                            injury_date="2024-11-24", surgery=True)
        report = adjuster.injury_report()
        assert len(report) == 1
        assert report.iloc[0]["player_name"] == "Tyreek Hill"
        assert report.iloc[0]["surgery"] is True
