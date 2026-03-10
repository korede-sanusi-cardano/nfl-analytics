"""
Injury Adjustment Model.

Adjusts Dynasty VORP projections based on injury severity, recovery
timelines, and positional impact. Without this, the model treats a
player coming off a broken leg the same as a fully healthy player —
which is how you end up overvaluing Tyreek Hill post-fracture.

Two adjustment mechanisms:
1. Games-played projection — reduces expected games in recovery season(s)
2. Severity multiplier — discounts the aging-curve retention factor
   during recovery, reflecting reduced burst/efficiency post-injury

Data sources (in priority order):
1. Sleeper API — real-time injury designations (IR, Out, Doubtful, etc.)
2. Web scraping — ESPN/PFR injury reports for injury type classification
3. Manual overrides — user-specified adjustments for edge cases

Usage:
    from src.models.injury_model import InjuryAdjuster
    from src.data.sleeper_client import SleeperClient

    adjuster = InjuryAdjuster()

    # Add known injuries
    adjuster.add_injury("Tyreek Hill", "WR", "broken_leg", severity="major",
                        injury_date="2024-11-24", surgery=True)

    # Pull Sleeper IR/injury designations automatically
    client = SleeperClient(league_id="1311316930342195200")
    adjuster.load_sleeper_injuries(client)

    # Apply to Dynasty VORP DataFrame
    adjusted_df = adjuster.adjust_dynasty_vorp(dynasty_vorp_df)
"""

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
from loguru import logger


# ──────────────────────────────────────────────────────────────────────
# Injury classification and impact tables
# ──────────────────────────────────────────────────────────────────────

@dataclass
class InjuryRecord:
    """A single injury event for a player."""

    player_name: str
    position: str
    injury_type: str  # key into INJURY_CATALOG
    severity: str = "moderate"  # minor, moderate, major, career_threatening
    injury_date: str | None = None  # ISO format YYYY-MM-DD
    surgery: bool = False
    source: str = "manual"  # manual, sleeper, web
    notes: str = ""

    @property
    def months_since_injury(self) -> float | None:
        if not self.injury_date:
            return None
        try:
            inj = datetime.strptime(self.injury_date, "%Y-%m-%d").date()
            delta = date.today() - inj
            return delta.days / 30.44
        except ValueError:
            return None


# Injury catalog: maps injury types to expected recovery and impact.
#
# games_missed: expected games missed in the season the injury occurred
#               or the following season if late-season injury
# retention_multiplier: applied to the aging-curve retention factor
#                       during recovery year(s). 1.0 = no impact, 0.5 = halved
# recovery_seasons: how many seasons the multiplier applies
# positional_modifier: positions where this injury has outsized impact
#                      (e.g., lower body injuries hit speed players harder)

INJURY_CATALOG = {
    # ── Lower body (structural) ──
    "acl": {
        "label": "ACL Tear",
        "games_missed": {"minor": 8, "moderate": 12, "major": 17, "career_threatening": 17},
        "retention_multiplier": {"minor": 0.85, "moderate": 0.75, "major": 0.65, "career_threatening": 0.50},
        "recovery_seasons": 2,
        "positional_modifier": {"RB": 0.90, "WR": 0.92, "TE": 0.95, "QB": 0.97},
    },
    "broken_leg": {
        "label": "Leg Fracture (Tibia/Fibula/Femur)",
        "games_missed": {"minor": 8, "moderate": 13, "major": 17, "career_threatening": 17},
        "retention_multiplier": {"minor": 0.80, "moderate": 0.70, "major": 0.60, "career_threatening": 0.45},
        "recovery_seasons": 2,
        "positional_modifier": {"RB": 0.88, "WR": 0.90, "TE": 0.93, "QB": 0.96},
    },
    "achilles": {
        "label": "Achilles Rupture",
        "games_missed": {"minor": 12, "moderate": 17, "major": 17, "career_threatening": 17},
        "retention_multiplier": {"minor": 0.75, "moderate": 0.65, "major": 0.55, "career_threatening": 0.40},
        "recovery_seasons": 2,
        "positional_modifier": {"RB": 0.85, "WR": 0.88, "TE": 0.92, "QB": 0.97},
    },
    "ankle": {
        "label": "Ankle Sprain/Fracture",
        "games_missed": {"minor": 2, "moderate": 5, "major": 10, "career_threatening": 14},
        "retention_multiplier": {"minor": 0.95, "moderate": 0.88, "major": 0.78, "career_threatening": 0.65},
        "recovery_seasons": 1,
        "positional_modifier": {"RB": 0.93, "WR": 0.94, "TE": 0.96, "QB": 0.98},
    },
    "knee_other": {
        "label": "Knee (Non-ACL: MCL/Meniscus)",
        "games_missed": {"minor": 2, "moderate": 6, "major": 10, "career_threatening": 15},
        "retention_multiplier": {"minor": 0.93, "moderate": 0.85, "major": 0.75, "career_threatening": 0.60},
        "recovery_seasons": 1,
        "positional_modifier": {"RB": 0.92, "WR": 0.94, "TE": 0.95, "QB": 0.97},
    },
    "foot": {
        "label": "Foot Injury (Lisfranc/Jones Fracture)",
        "games_missed": {"minor": 4, "moderate": 10, "major": 17, "career_threatening": 17},
        "retention_multiplier": {"minor": 0.90, "moderate": 0.78, "major": 0.65, "career_threatening": 0.50},
        "recovery_seasons": 2,
        "positional_modifier": {"RB": 0.88, "WR": 0.90, "TE": 0.93, "QB": 0.96},
    },
    # ── Upper body ──
    "shoulder": {
        "label": "Shoulder (Dislocation/Labrum/Rotator Cuff)",
        "games_missed": {"minor": 2, "moderate": 6, "major": 12, "career_threatening": 17},
        "retention_multiplier": {"minor": 0.95, "moderate": 0.88, "major": 0.78, "career_threatening": 0.65},
        "recovery_seasons": 1,
        "positional_modifier": {"RB": 0.97, "WR": 0.93, "TE": 0.93, "QB": 0.90},
    },
    "hand_wrist": {
        "label": "Hand/Wrist Fracture",
        "games_missed": {"minor": 1, "moderate": 4, "major": 8, "career_threatening": 12},
        "retention_multiplier": {"minor": 0.97, "moderate": 0.92, "major": 0.85, "career_threatening": 0.75},
        "recovery_seasons": 1,
        "positional_modifier": {"RB": 0.98, "WR": 0.93, "TE": 0.94, "QB": 0.90},
    },
    # ── Head / Neck ──
    "concussion": {
        "label": "Concussion",
        "games_missed": {"minor": 1, "moderate": 3, "major": 6, "career_threatening": 17},
        "retention_multiplier": {"minor": 0.97, "moderate": 0.92, "major": 0.82, "career_threatening": 0.60},
        "recovery_seasons": 1,
        "positional_modifier": {"RB": 0.96, "WR": 0.96, "TE": 0.96, "QB": 0.95},
    },
    "neck": {
        "label": "Neck/Spine Injury",
        "games_missed": {"minor": 2, "moderate": 8, "major": 17, "career_threatening": 17},
        "retention_multiplier": {"minor": 0.90, "moderate": 0.75, "major": 0.55, "career_threatening": 0.30},
        "recovery_seasons": 2,
        "positional_modifier": {"RB": 0.90, "WR": 0.92, "TE": 0.92, "QB": 0.90},
    },
    # ── Soft tissue ──
    "hamstring": {
        "label": "Hamstring Strain/Tear",
        "games_missed": {"minor": 1, "moderate": 4, "major": 8, "career_threatening": 12},
        "retention_multiplier": {"minor": 0.96, "moderate": 0.90, "major": 0.82, "career_threatening": 0.70},
        "recovery_seasons": 1,
        "positional_modifier": {"RB": 0.93, "WR": 0.92, "TE": 0.95, "QB": 0.97},
    },
    "quad_calf": {
        "label": "Quad/Calf Strain",
        "games_missed": {"minor": 1, "moderate": 3, "major": 6, "career_threatening": 10},
        "retention_multiplier": {"minor": 0.97, "moderate": 0.92, "major": 0.85, "career_threatening": 0.75},
        "recovery_seasons": 1,
        "positional_modifier": {"RB": 0.94, "WR": 0.93, "TE": 0.96, "QB": 0.98},
    },
    # ── General / Unknown ──
    "general_ir": {
        "label": "Placed on IR (Unspecified)",
        "games_missed": {"minor": 4, "moderate": 8, "major": 12, "career_threatening": 17},
        "retention_multiplier": {"minor": 0.92, "moderate": 0.85, "major": 0.75, "career_threatening": 0.60},
        "recovery_seasons": 1,
        "positional_modifier": {"RB": 0.95, "WR": 0.95, "TE": 0.96, "QB": 0.96},
    },
}

# Sleeper API injury status → rough severity mapping
SLEEPER_STATUS_MAP = {
    "IR": ("general_ir", "major"),
    "Out": ("general_ir", "moderate"),
    "Doubtful": ("general_ir", "minor"),
    "Questionable": (None, None),  # too mild to adjust
    "Probable": (None, None),
    "Sus": ("general_ir", "moderate"),  # suspended, treat as missed time
    "PUP": ("general_ir", "moderate"),
    "NFI": ("general_ir", "moderate"),
    "COV": (None, None),  # COVID list, usually short
}

# Age threshold where injuries become more impactful
# After this age, the retention multiplier gets an extra penalty
AGE_INJURY_PENALTY = {
    "QB": 34,
    "RB": 27,
    "WR": 30,
    "TE": 30,
}


# ──────────────────────────────────────────────────────────────────────
# Main adjuster class
# ──────────────────────────────────────────────────────────────────────

class InjuryAdjuster:
    """
    Adjusts Dynasty VORP projections for player injuries.

    Maintains a registry of known injuries and applies two adjustments:
    1. Games-played projection for the recovery season
    2. Retention multiplier discount on the aging curve
    """

    def __init__(self):
        self._injuries: dict[str, list[InjuryRecord]] = {}  # player_name → injuries

    @property
    def injury_count(self) -> int:
        return sum(len(v) for v in self._injuries.values())

    # ── Data loading ──────────────────────────────────────────────────

    def add_injury(
        self,
        player_name: str,
        position: str,
        injury_type: str,
        severity: str = "moderate",
        injury_date: str | None = None,
        surgery: bool = False,
        notes: str = "",
        source: str = "manual",
    ) -> None:
        """Add a single injury record."""
        if injury_type not in INJURY_CATALOG:
            logger.warning(
                f"Unknown injury type '{injury_type}' for {player_name}. "
                f"Valid types: {list(INJURY_CATALOG.keys())}"
            )
            return

        record = InjuryRecord(
            player_name=player_name,
            position=position,
            injury_type=injury_type,
            severity=severity,
            injury_date=injury_date,
            surgery=surgery,
            source=source,
            notes=notes,
        )

        if player_name not in self._injuries:
            self._injuries[player_name] = []
        self._injuries[player_name].append(record)

        logger.info(
            f"Added injury: {player_name} — {INJURY_CATALOG[injury_type]['label']} "
            f"({severity}), source={source}"
        )

    def load_sleeper_injuries(self, client) -> int:
        """
        Pull injury designations from Sleeper's player database.

        Sleeper marks players with statuses like IR, Out, Doubtful, etc.
        We map these to our injury catalog with conservative severity estimates.
        Lower priority than manual entries — won't overwrite existing records.

        Returns count of injuries loaded.
        """
        try:
            all_players = client.get_all_players()
        except Exception as e:
            logger.warning(f"Could not load Sleeper player data: {e}")
            return 0

        count = 0
        for player_id, info in all_players.items():
            status = info.get("injury_status") or info.get("status")
            if not status or status not in SLEEPER_STATUS_MAP:
                continue

            injury_type, severity = SLEEPER_STATUS_MAP[status]
            if injury_type is None:
                continue

            name = info.get("full_name", "")
            position = info.get("position", "")
            if not name or position not in ("QB", "RB", "WR", "TE"):
                continue

            # Don't overwrite manual entries (higher priority)
            if name in self._injuries:
                existing_sources = {r.source for r in self._injuries[name]}
                if "manual" in existing_sources:
                    continue

            self.add_injury(
                player_name=name,
                position=position,
                injury_type=injury_type,
                severity=severity,
                source="sleeper",
                notes=f"Sleeper status: {status}",
            )
            count += 1

        logger.info(f"Loaded {count} injuries from Sleeper API")
        return count

    def load_manual_overrides(self, overrides: list[dict]) -> int:
        """
        Load a batch of manual injury overrides.

        Each dict should have: player_name, position, injury_type, severity,
        and optionally: injury_date, surgery, notes.

        Example:
            adjuster.load_manual_overrides([
                {"player_name": "Tyreek Hill", "position": "WR",
                 "injury_type": "broken_leg", "severity": "major",
                 "injury_date": "2024-11-24", "surgery": True},
            ])
        """
        count = 0
        for entry in overrides:
            try:
                self.add_injury(source="manual", **entry)
                count += 1
            except TypeError as e:
                logger.warning(f"Invalid override entry: {entry} — {e}")
        return count

    # ── Adjustment calculations ───────────────────────────────────────

    def get_games_projection(
        self,
        player_name: str,
        position: str,
        base_games: int = 17,
    ) -> int:
        """
        Project games played in the recovery season.

        If a player has multiple injuries, uses the most severe one.
        """
        injuries = self._injuries.get(player_name, [])
        if not injuries:
            return base_games

        # Use the most impactful injury
        worst = self._get_worst_injury(injuries)
        catalog = INJURY_CATALOG[worst.injury_type]
        games_missed = catalog["games_missed"].get(worst.severity, 8)

        # If injury was recent (< 6 months), full impact
        # If older, scale down the games missed
        months = worst.months_since_injury
        if months is not None and months > 10:
            # More than 10 months ago — likely recovered for next season
            games_missed = max(0, games_missed - 10)
        elif months is not None and months > 6:
            games_missed = int(games_missed * 0.5)

        # Surgery adds recovery time
        if worst.surgery:
            games_missed = min(17, int(games_missed * 1.2))

        projected = max(0, base_games - games_missed)
        return projected

    def get_retention_adjustment(
        self,
        player_name: str,
        position: str,
        age: int,
        projection_year: int = 1,
    ) -> float:
        """
        Get the injury-adjusted retention multiplier for a projection year.

        Returns a value between 0 and 1 that should be multiplied with
        the standard aging-curve retention factor.

        Args:
            player_name: Player to check
            position: QB/RB/WR/TE
            age: Player's current age
            projection_year: Which future year (1 = next season, 2 = two out, etc.)

        Returns:
            Multiplier to apply to retention (1.0 = no adjustment)
        """
        injuries = self._injuries.get(player_name, [])
        if not injuries:
            return 1.0

        worst = self._get_worst_injury(injuries)
        catalog = INJURY_CATALOG[worst.injury_type]
        recovery_seasons = catalog["recovery_seasons"]

        # No adjustment if we're past the recovery window
        if projection_year > recovery_seasons:
            return 1.0

        # Base retention multiplier from catalog
        base_mult = catalog["retention_multiplier"].get(worst.severity, 0.85)

        # Positional modifier (speed players hit harder by lower body injuries)
        pos_mod = catalog["positional_modifier"].get(position, 0.95)

        # Age penalty: older players recover worse
        age_threshold = AGE_INJURY_PENALTY.get(position, 30)
        age_penalty = 1.0
        if age > age_threshold:
            years_over = age - age_threshold
            age_penalty = max(0.70, 1.0 - years_over * 0.04)

        # Surgery penalty
        surgery_penalty = 0.92 if worst.surgery else 1.0

        # Year 2 of recovery is less impactful than year 1
        year_decay = 1.0 if projection_year == 1 else 0.5

        # Combine: base multiplier, positional, age, surgery
        # The year_decay interpolates back toward 1.0 for later recovery years
        adjustment = base_mult * pos_mod * age_penalty * surgery_penalty
        final = 1.0 - (1.0 - adjustment) * (1.0 if year_decay == 1.0 else year_decay)

        return max(0.20, min(1.0, final))

    def _get_worst_injury(self, injuries: list[InjuryRecord]) -> InjuryRecord:
        """Return the most severe injury from a list."""
        severity_order = {"career_threatening": 4, "major": 3, "moderate": 2, "minor": 1}
        return max(injuries, key=lambda r: severity_order.get(r.severity, 0))

    # ── Apply to Dynasty VORP ─────────────────────────────────────────

    def adjust_dynasty_vorp(
        self,
        dynasty_df: pd.DataFrame,
        projection_years: int = 5,
        discount_rate: float = 0.10,
    ) -> pd.DataFrame:
        """
        Apply injury adjustments to an existing Dynasty VORP DataFrame.

        Recalculates dynasty_vorp by:
        1. Reducing games_played for recovery season
        2. Applying retention multipliers to future projections

        The input DataFrame should already have vorp and dynasty_vorp
        columns from VORPCalculator.calculate_dynasty_vorp().

        Returns a new DataFrame with:
        - injury_adjusted_vorp: the corrected dynasty VORP
        - injury_status: description of the injury
        - games_projected: expected games in recovery season
        - injury_discount: the total discount applied (as a %)
        """
        df = dynasty_df.copy()

        adjusted_vorps = []
        injury_statuses = []
        games_projected = []
        injury_discounts = []

        for _, row in df.iterrows():
            name = row.get("player_name", "")
            position = row.get("position", "")
            age = row.get("age", 26) if pd.notna(row.get("age")) else 26
            current_vorp = row.get("vorp", 0)

            injuries = self._injuries.get(name, [])

            if not injuries:
                # No injury — keep original dynasty VORP
                adjusted_vorps.append(row.get("dynasty_vorp", current_vorp))
                injury_statuses.append("Healthy")
                games_projected.append(17)
                injury_discounts.append(0.0)
                continue

            worst = self._get_worst_injury(injuries)
            catalog = INJURY_CATALOG.get(worst.injury_type, {})
            status = f"{catalog.get('label', worst.injury_type)} ({worst.severity})"
            if worst.surgery:
                status += " [Surgery]"

            # Projected games in year 1
            yr1_games = self.get_games_projection(name, position)
            game_fraction = yr1_games / 17.0

            # Recalculate dynasty VORP with injury adjustments
            # Year 0: current season, scaled by games fraction
            adj_dynasty = current_vorp * game_fraction
            projected_vorp = current_vorp

            from src.models.vorp import get_retention_factor

            for year in range(1, projection_years + 1):
                future_age = age + year
                retention = get_retention_factor(position, future_age)

                # Apply injury retention adjustment
                injury_adj = self.get_retention_adjustment(
                    name, position, age, projection_year=year
                )
                retention *= injury_adj

                projected_vorp *= retention
                discounted = projected_vorp / ((1 + discount_rate) ** year)
                adj_dynasty += discounted

            original = row.get("dynasty_vorp", current_vorp)
            discount_pct = (
                round((1 - adj_dynasty / original) * 100, 1)
                if original > 0 else 0.0
            )

            adjusted_vorps.append(round(adj_dynasty, 2))
            injury_statuses.append(status)
            games_projected.append(yr1_games)
            injury_discounts.append(discount_pct)

        df["injury_adjusted_vorp"] = adjusted_vorps
        df["injury_status"] = injury_statuses
        df["games_projected"] = games_projected
        df["injury_discount_pct"] = injury_discounts

        # Re-rank by injury-adjusted VORP
        df["injury_adj_rank"] = (
            df["injury_adjusted_vorp"].rank(ascending=False).astype(int)
        )

        logger.info(
            f"Applied injury adjustments to {len(df)} players. "
            f"{self.injury_count} injuries tracked."
        )

        return df.sort_values("injury_adjusted_vorp", ascending=False).reset_index(drop=True)

    # ── Reporting ─────────────────────────────────────────────────────

    def injury_report(self) -> pd.DataFrame:
        """Generate a summary of all tracked injuries."""
        rows = []
        for name, injuries in self._injuries.items():
            for inj in injuries:
                catalog = INJURY_CATALOG.get(inj.injury_type, {})
                rows.append({
                    "player_name": name,
                    "position": inj.position,
                    "injury": catalog.get("label", inj.injury_type),
                    "severity": inj.severity,
                    "surgery": inj.surgery,
                    "injury_date": inj.injury_date,
                    "months_ago": round(inj.months_since_injury, 1) if inj.months_since_injury else None,
                    "source": inj.source,
                    "notes": inj.notes,
                })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values("severity", ascending=False)
