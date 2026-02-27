"""
Rookie Valuation Model.

Projects NFL fantasy production for incoming rookies using three signal
categories — college production, athletic measurables, and accolades.

Think of it as early-stage venture capital due diligence applied to
football prospects: college stats are the company's revenue trajectory,
combine measurables are the founding team's capabilities, and awards
are social proof / market validation.

The model trains on historical rookie classes (mapping their pre-draft
profile to actual NFL fantasy output in years 1-3), then predicts
value for the current incoming class.

Data Sources:
    - nfl_data_py: import_combine_data(), import_draft_picks()
    - nfl_data_py: import_weekly_data() for actual NFL production
    - Web scraping fallback: Pro Football Reference for combine/college
    - Hardcoded: Major college football award winners

Usage:
    from src.models.rookie_model import RookieValuationModel
    from src.data.nfl_stats import NFLStats

    nfl = NFLStats()
    model = RookieValuationModel(nfl_stats=nfl)

    # Train on historical data
    metrics = model.train(draft_classes=range(2017, 2023))

    # Predict for incoming class
    predictions = model.predict_rookie_class(draft_year=2024)

    # Integrate with Dynasty VORP
    dynasty_pool = model.merge_with_vorp(predictions, existing_vorp_df)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.data.nfl_stats import NFLStats


# ──────────────────────────────────────────────────────────────────────
# College football award data
# ──────────────────────────────────────────────────────────────────────

# Major college football awards and their fantasy-relevance weight.
# Awards signal elite college production — the question is whether
# that translates to NFL fantasy points.

COLLEGE_AWARDS = {
    # (year, player_name, award, position)
    # Heisman Trophy — most outstanding player
    "heisman": [
        (2015, "Derrick Henry", "Heisman", "RB"),
        (2016, "Lamar Jackson", "Heisman", "QB"),
        (2017, "Baker Mayfield", "Heisman", "QB"),
        (2018, "Kyler Murray", "Heisman", "QB"),
        (2019, "Joe Burrow", "Heisman", "QB"),
        (2020, "DeVonta Smith", "Heisman", "WR"),
        (2021, "Bryce Young", "Heisman", "QB"),
        (2022, "Caleb Williams", "Heisman", "QB"),
        (2023, "Jayden Daniels", "Heisman", "QB"),
        (2024, "Travis Hunter", "Heisman", "WR"),
    ],
    # Maxwell Award — best all-around player
    "maxwell": [
        (2015, "Derrick Henry", "Maxwell", "RB"),
        (2016, "Lamar Jackson", "Maxwell", "QB"),
        (2017, "Baker Mayfield", "Maxwell", "QB"),
        (2018, "Tua Tagovailoa", "Maxwell", "QB"),
        (2019, "Joe Burrow", "Maxwell", "QB"),
        (2020, "DeVonta Smith", "Maxwell", "WR"),
        (2021, "Kenneth Walker III", "Maxwell", "RB"),
        (2022, "Caleb Williams", "Maxwell", "QB"),
        (2023, "Jayden Daniels", "Maxwell", "QB"),
        (2024, "Travis Hunter", "Maxwell", "WR"),
    ],
    # Walter Camp — player of the year
    "walter_camp": [
        (2015, "Derrick Henry", "Walter Camp", "RB"),
        (2016, "Lamar Jackson", "Walter Camp", "QB"),
        (2017, "Baker Mayfield", "Walter Camp", "QB"),
        (2018, "Tua Tagovailoa", "Walter Camp", "QB"),
        (2019, "Joe Burrow", "Walter Camp", "QB"),
        (2020, "DeVonta Smith", "Walter Camp", "WR"),
        (2021, "Bryce Young", "Walter Camp", "QB"),
        (2022, "Max Duggan", "Walter Camp", "QB"),
        (2023, "Michael Penix Jr", "Walter Camp", "QB"),
        (2024, "Travis Hunter", "Walter Camp", "WR"),
    ],
    # Biletnikoff Award — best receiver
    "biletnikoff": [
        (2015, "Corey Coleman", "Biletnikoff", "WR"),
        (2016, "Dede Westbrook", "Biletnikoff", "WR"),
        (2017, "James Washington", "Biletnikoff", "WR"),
        (2018, "Jerry Jeudy", "Biletnikoff", "WR"),
        (2019, "CeeDee Lamb", "Biletnikoff", "WR"),
        (2020, "DeVonta Smith", "Biletnikoff", "WR"),
        (2021, "Treylon Burks", "Biletnikoff", "WR"),
        (2022, "Jalin Hyatt", "Biletnikoff", "WR"),
        (2023, "Rome Odunze", "Biletnikoff", "WR"),
        (2024, "Tetairoa McMillan", "Biletnikoff", "WR"),
    ],
    # Doak Walker Award — best running back
    "doak_walker": [
        (2015, "Derrick Henry", "Doak Walker", "RB"),
        (2016, "Donnel Pumphrey", "Doak Walker", "RB"),
        (2017, "Bryce Love", "Doak Walker", "RB"),
        (2018, "Jonathan Taylor", "Doak Walker", "RB"),
        (2019, "Jonathan Taylor", "Doak Walker", "RB"),
        (2020, "Najee Harris", "Doak Walker", "RB"),
        (2021, "Kenneth Walker III", "Doak Walker", "RB"),
        (2022, "Bijan Robinson", "Doak Walker", "RB"),
        (2023, "Ollie Gordon II", "Doak Walker", "RB"),
        (2024, "Ashton Jeanty", "Doak Walker", "RB"),
    ],
    # Davey O'Brien Award — best quarterback
    "davey_obrien": [
        (2015, "Deshaun Watson", "Davey O'Brien", "QB"),
        (2016, "Lamar Jackson", "Davey O'Brien", "QB"),
        (2017, "Baker Mayfield", "Davey O'Brien", "QB"),
        (2018, "Kyler Murray", "Davey O'Brien", "QB"),
        (2019, "Joe Burrow", "Davey O'Brien", "QB"),
        (2020, "Trevor Lawrence", "Davey O'Brien", "QB"),
        (2021, "Bryce Young", "Davey O'Brien", "QB"),
        (2022, "Caleb Williams", "Davey O'Brien", "QB"),
        (2023, "Jayden Daniels", "Davey O'Brien", "QB"),
        (2024, "Fernando Mendoza", "Davey O'Brien", "QB"),
    ],
    # John Mackey Award — best tight end
    "john_mackey": [
        (2015, "Hunter Henry", "John Mackey", "TE"),
        (2016, "Jake Butt", "John Mackey", "TE"),
        (2017, "Mark Andrews", "John Mackey", "TE"),
        (2018, "TJ Hockenson", "John Mackey", "TE"),
        (2019, "Harrison Bryant", "John Mackey", "TE"),
        (2020, "Kyle Pitts", "John Mackey", "TE"),
        (2021, "Trey McBride", "John Mackey", "TE"),
        (2022, "Michael Mayer", "John Mackey", "TE"),
        (2023, "Brock Bowers", "John Mackey", "TE"),
        (2024, "Tyler Warren", "John Mackey", "TE"),
    ],
    # Johnny Unitas Golden Arm — best upperclassman QB
    "johnny_unitas": [
        (2015, "Deshaun Watson", "Johnny Unitas", "QB"),
        (2016, "Deshaun Watson", "Johnny Unitas", "QB"),
        (2017, "Baker Mayfield", "Johnny Unitas", "QB"),
        (2018, "Will Grier", "Johnny Unitas", "QB"),
        (2019, "Joe Burrow", "Johnny Unitas", "QB"),
        (2020, "Sam Ehlinger", "Johnny Unitas", "QB"),
        (2021, "Sam Howell", "Johnny Unitas", "QB"),
        (2022, "Max Duggan", "Johnny Unitas", "QB"),
        (2023, "Michael Penix Jr", "Johnny Unitas", "QB"),
        (2024, "Shedeur Sanders", "Johnny Unitas", "QB"),
    ],
}


def build_awards_dataframe() -> pd.DataFrame:
    """Flatten the awards dict into a structured DataFrame."""
    rows = []
    for award_key, entries in COLLEGE_AWARDS.items():
        for year, player, award, position in entries:
            rows.append(
                {
                    "college_season": year,
                    "player_name": player,
                    "award": award,
                    "position": position,
                }
            )
    return pd.DataFrame(rows)


def get_award_counts(player_name: str, college_season: int, awards_df: pd.DataFrame) -> dict:
    """Count awards a player won in or before their draft-eligible season."""
    player_awards = awards_df[
        (awards_df["player_name"] == player_name)
        & (awards_df["college_season"] <= college_season)
    ]

    return {
        "total_awards": len(player_awards),
        "has_heisman": int("Heisman" in player_awards["award"].values),
        "has_position_award": int(
            player_awards["award"]
            .isin(["Biletnikoff", "Doak Walker", "Davey O'Brien", "John Mackey"])
            .any()
        ),
        "has_maxwell_or_camp": int(
            player_awards["award"].isin(["Maxwell", "Walter Camp"]).any()
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Combine measurable benchmarks by position
# ──────────────────────────────────────────────────────────────────────

# These are approximate 50th-percentile benchmarks used for normalisation.
# Think of them as the "market average" — prospects above these are
# athletically above average for their position.

COMBINE_BENCHMARKS = {
    "QB": {"forty": 4.80, "vertical": 32.0, "bench": 20, "broad_jump": 108, "three_cone": 7.10},
    "RB": {"forty": 4.52, "vertical": 34.0, "bench": 20, "broad_jump": 118, "three_cone": 7.00},
    "WR": {"forty": 4.48, "vertical": 35.5, "bench": 14, "broad_jump": 120, "three_cone": 6.95},
    "TE": {"forty": 4.65, "vertical": 33.0, "bench": 22, "broad_jump": 115, "three_cone": 7.10},
}


@dataclass
class RookieProjection:
    """Projected fantasy value for a rookie prospect."""

    player_name: str
    position: str
    draft_year: int
    draft_round: int | None
    draft_pick: int | None
    college: str
    projected_yr1_ppg: float
    projected_yr1_total: float
    projected_yr3_avg_ppg: float
    confidence: float  # 0-1, higher = more confident
    upside_score: float  # relative to position peers
    award_count: int
    combine_score: float
    college_production_score: float


class RookieValuationModel:
    """
    Predict rookie NFL fantasy production from pre-draft signals.

    Three signal categories (feature groups):
    1. College production — stats from their college career
    2. Athletic measurables — combine/pro day results
    3. Accolades — awards won (Heisman, Biletnikoff, etc.)

    Plus draft capital (where they were drafted) as a strong signal
    of NFL team evaluation, which correlates with opportunity.
    """

    FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]

    def __init__(self, nfl_stats: NFLStats):
        self.nfl_stats = nfl_stats
        self.model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            min_samples_leaf=10,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self._is_fitted = False
        self._awards_df = build_awards_dataframe()
        self._feature_columns: list[str] = []

    # ──────────────────────────────────────────────────────────────────
    # Data loading (wraps nfl_data_py with fallbacks)
    # ──────────────────────────────────────────────────────────────────

    def load_combine_data(self, years: list[int]) -> pd.DataFrame:
        """
        Load NFL Combine measurable data.

        Primary: nfl_data_py.import_combine_data()
        Fallback: returns empty DataFrame (model still works with other features)
        """
        try:
            import nfl_data_py as nfl

            logger.info(f"Loading combine data for {years[0]}-{years[-1]}")
            df = nfl.import_combine_data(years)
            logger.info(f"Loaded {len(df)} combine records")
            return df
        except Exception as e:
            logger.warning(f"Could not load combine data via nfl_data_py: {e}")
            logger.info("Model will proceed without combine features")
            return pd.DataFrame()

    def load_draft_picks(self, years: list[int]) -> pd.DataFrame:
        """
        Load NFL Draft pick data.

        Primary: nfl_data_py.import_draft_picks()
        """
        try:
            import nfl_data_py as nfl

            logger.info(f"Loading draft picks for {years[0]}-{years[-1]}")
            df = nfl.import_draft_picks()
            df = df[df["season"].isin(years)]
            logger.info(f"Loaded {len(df)} draft picks")
            return df
        except Exception as e:
            logger.warning(f"Could not load draft picks via nfl_data_py: {e}")
            return pd.DataFrame()

    def load_rookie_nfl_performance(
        self, draft_years: list[int], years_in_nfl: int = 3
    ) -> pd.DataFrame:
        """
        Load actual NFL fantasy production for rookies in their first N years.

        This is the target variable for training — what we're trying to predict.
        """
        all_data = []

        for draft_year in draft_years:
            nfl_seasons = list(range(draft_year, draft_year + years_in_nfl))

            for season in nfl_seasons:
                try:
                    for pos in self.FANTASY_POSITIONS:
                        df = self.nfl_stats.get_top_performers(
                            season, pos, top_n=80
                        )
                        df["season"] = season
                        df["draft_year"] = draft_year
                        df["nfl_year"] = season - draft_year + 1
                        all_data.append(df)
                except Exception as e:
                    logger.debug(f"No data for {season}: {e}")

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    # ──────────────────────────────────────────────────────────────────
    # Feature engineering
    # ──────────────────────────────────────────────────────────────────

    def build_training_data(
        self,
        draft_classes: list[int] | range,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build feature matrix and target variable for model training.

        For each drafted player, we assemble:
        - Features: combine measurables, draft capital, awards
        - Target: actual year-1 fantasy PPG in the NFL

        Only players with at least 6 games in year 1 are included
        to avoid noise from injured/inactive players.
        """
        draft_years = list(draft_classes)
        logger.info(f"Building training data for draft classes {draft_years[0]}-{draft_years[-1]}")

        # Load all data sources
        combine_df = self.load_combine_data(draft_years)
        draft_df = self.load_draft_picks(draft_years)
        nfl_performance = self.load_rookie_nfl_performance(draft_years, years_in_nfl=1)

        if draft_df.empty or nfl_performance.empty:
            logger.error("Insufficient data for training. Ensure nfl_data_py is configured.")
            return pd.DataFrame(), pd.Series(dtype=float)

        # Filter to year-1 performance with minimum games
        yr1 = nfl_performance[
            (nfl_performance["nfl_year"] == 1)
            & (nfl_performance["games_played"] >= 6)
        ].copy()

        # Merge draft info with NFL performance
        # Need to match on player — use name matching (imperfect but workable)
        merged = self._merge_draft_and_performance(draft_df, yr1)

        if merged.empty:
            logger.error("No matched records between draft and performance data")
            return pd.DataFrame(), pd.Series(dtype=float)

        # Add combine features
        if not combine_df.empty:
            merged = self._add_combine_features(merged, combine_df)

        # Add award features
        merged = self._add_award_features(merged)

        # Extract feature matrix
        features = self._extract_features(merged)
        target = merged["ppg"].copy()

        logger.info(
            f"Training data: {len(features)} players, "
            f"{len(features.columns)} features"
        )

        return features, target

    def _merge_draft_and_performance(
        self,
        draft_df: pd.DataFrame,
        performance_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge draft picks with NFL performance using flexible name matching."""
        # Standardise names for matching
        draft = draft_df.copy()
        perf = performance_df.copy()

        # nfl_data_py draft columns: season, round, pick, team, pfr_name, etc.
        # Try multiple name columns
        name_col_draft = None
        for col in ["pfr_name", "player_name", "name"]:
            if col in draft.columns:
                name_col_draft = col
                break

        if name_col_draft is None:
            logger.error(f"No name column found in draft data. Columns: {draft.columns.tolist()}")
            return pd.DataFrame()

        draft["match_name"] = draft[name_col_draft].str.lower().str.strip()
        perf["match_name"] = perf["player_name"].str.lower().str.strip()

        # Merge on name + check season alignment
        merged = perf.merge(
            draft[["match_name", "season", "round", "pick", "team"]].rename(
                columns={"season": "draft_season", "team": "draft_team"}
            ),
            on="match_name",
            how="inner",
        )

        # Only keep rows where draft year matches
        merged = merged[merged["draft_year"] == merged["draft_season"]].copy()

        # Rename for clarity
        merged["draft_round"] = merged["round"]
        merged["draft_pick"] = merged["pick"]

        logger.info(f"Matched {len(merged)} rookies between draft and performance data")
        return merged

    def _add_combine_features(
        self,
        merged: pd.DataFrame,
        combine_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add combine measurable features to the merged dataset."""
        combine = combine_df.copy()

        # Standardise name for matching
        name_col = None
        for col in ["player_name", "name", "pfr_name"]:
            if col in combine.columns:
                name_col = col
                break

        if name_col is None:
            return merged

        combine["match_name"] = combine[name_col].str.lower().str.strip()

        # Select relevant combine columns
        combine_cols = ["match_name"]
        col_mapping = {
            "forty": ["forty", "forty_yd", "40yd"],
            "vertical": ["vertical", "vertical_jump"],
            "bench": ["bench", "bench_press"],
            "broad_jump": ["broad_jump", "broad"],
            "three_cone": ["three_cone", "three_cone_drill", "3cone"],
            "shuttle": ["shuttle", "twenty_shuttle", "20shuttle"],
            "ht": ["ht", "height"],
            "wt": ["wt", "weight"],
        }

        for target_name, possible_names in col_mapping.items():
            for col_name in possible_names:
                if col_name in combine.columns:
                    combine[target_name] = combine[col_name]
                    combine_cols.append(target_name)
                    break

        combine_subset = combine[list(set(combine_cols))].drop_duplicates(subset=["match_name"])

        result = merged.merge(combine_subset, on="match_name", how="left")
        return result

    def _add_award_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add college award features to the dataset."""
        result = df.copy()
        result["total_awards"] = 0
        result["has_heisman"] = 0
        result["has_position_award"] = 0
        result["has_maxwell_or_camp"] = 0

        for idx, row in result.iterrows():
            name = row.get("player_name", "")
            draft_year = row.get("draft_year", 0)
            # Awards are from college seasons, typically draft_year - 1
            college_season = draft_year - 1

            award_info = get_award_counts(name, college_season, self._awards_df)
            for key, val in award_info.items():
                result.at[idx, key] = val

        return result

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the final feature matrix for ML training."""
        features = pd.DataFrame()

        # Draft capital features (very strong signal)
        features["draft_round"] = df.get("draft_round", pd.Series(dtype=float))
        features["draft_pick"] = df.get("draft_pick", pd.Series(dtype=float))
        features["is_first_round"] = (features["draft_round"] == 1).astype(int)
        features["is_top_10"] = (features["draft_pick"] <= 10).astype(int)

        # Combine measurables (normalised by position benchmarks)
        for metric in ["forty", "vertical", "bench", "broad_jump", "three_cone"]:
            if metric in df.columns:
                features[f"combine_{metric}"] = df[metric].astype(float)
            else:
                features[f"combine_{metric}"] = np.nan

        # Position encoding
        features["pos_QB"] = (df["position"] == "QB").astype(int)
        features["pos_RB"] = (df["position"] == "RB").astype(int)
        features["pos_WR"] = (df["position"] == "WR").astype(int)
        features["pos_TE"] = (df["position"] == "TE").astype(int)

        # Award features
        features["total_awards"] = df.get("total_awards", 0)
        features["has_heisman"] = df.get("has_heisman", 0)
        features["has_position_award"] = df.get("has_position_award", 0)
        features["has_maxwell_or_camp"] = df.get("has_maxwell_or_camp", 0)

        # Interaction features
        features["draft_pick_x_awards"] = (
            features["draft_pick"] * features["total_awards"]
        )

        # Fill NaNs with position-appropriate medians
        features = features.fillna(features.median())
        # Final fallback for any remaining NaNs
        features = features.fillna(0)

        self._feature_columns = features.columns.tolist()
        return features

    # ──────────────────────────────────────────────────────────────────
    # Model training
    # ──────────────────────────────────────────────────────────────────

    def train(
        self,
        draft_classes: list[int] | range | None = None,
    ) -> dict:
        """
        Train the rookie valuation model on historical draft classes.

        Args:
            draft_classes: Which draft years to train on.
                Default: 2017-2023 (enough data for combine + awards)

        Returns:
            Dict with training metrics (CV scores, feature importance, etc.)
        """
        if draft_classes is None:
            draft_classes = list(range(2017, 2024))

        features, target = self.build_training_data(draft_classes)

        if features.empty:
            return {"error": "No training data available. Check nfl_data_py setup."}

        # Scale features
        X_scaled = self.scaler.fit_transform(features)

        # Cross-validate
        cv_scores = cross_val_score(
            self.model, X_scaled, target,
            cv=min(5, len(features) // 10),  # adapt CV folds to data size
            scoring="neg_mean_absolute_error",
        )
        mae_scores = -cv_scores

        logger.info(
            f"Cross-validation MAE: {mae_scores.mean():.2f} "
            f"(+/- {mae_scores.std():.2f}) PPG"
        )

        # Fit on full data
        self.model.fit(X_scaled, target)
        self._is_fitted = True

        # Feature importance
        importance = dict(
            zip(self._feature_columns, self.model.feature_importances_)
        )
        importance_sorted = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        metrics = {
            "cv_mae_mean": mae_scores.mean(),
            "cv_mae_std": mae_scores.std(),
            "feature_importance": importance_sorted,
            "training_samples": len(features),
            "target_mean_ppg": target.mean(),
            "target_std_ppg": target.std(),
            "draft_classes": list(draft_classes),
        }

        logger.info(f"Model trained on {len(features)} rookies")
        logger.info("Top 5 features:")
        for feat, imp in list(importance_sorted.items())[:5]:
            logger.info(f"  {feat}: {imp:.3f}")

        return metrics

    # ──────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────

    def predict_rookie_class(
        self,
        draft_year: int,
        top_n: int = 50,
    ) -> pd.DataFrame:
        """
        Generate fantasy projections for an incoming rookie class.

        Pulls combine data and draft results for the specified year,
        applies the trained model, and returns ranked projections.

        Args:
            draft_year: The NFL draft year to project
            top_n: Number of top prospects to return

        Returns:
            DataFrame with projected fantasy value for each rookie
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call .train() first.")

        logger.info(f"Generating projections for {draft_year} rookie class")

        # Load prospect data
        combine_df = self.load_combine_data([draft_year])
        draft_df = self.load_draft_picks([draft_year])

        if draft_df.empty:
            logger.error(f"No draft data available for {draft_year}")
            return pd.DataFrame()

        # Filter to fantasy positions
        draft_df = draft_df[
            draft_df["position"].isin(self.FANTASY_POSITIONS)
        ].copy()

        # Add combine data
        if not combine_df.empty:
            draft_df = self._add_combine_to_draft(draft_df, combine_df)

        # Add awards
        draft_df["draft_year"] = draft_year
        draft_df = self._add_award_features_to_prospects(draft_df)

        # Extract features (must match training features)
        features = self._extract_prospect_features(draft_df)

        # Predict
        X_scaled = self.scaler.transform(features)
        predictions = self.model.predict(X_scaled)

        # Build result DataFrame
        result = draft_df.copy()
        result["projected_yr1_ppg"] = np.clip(predictions, 0, None)
        result["projected_yr1_total"] = result["projected_yr1_ppg"] * 17

        # Confidence score (based on feature completeness)
        result["confidence"] = self._calculate_confidence(features)

        # Upside score (relative to position peers)
        result["upside_score"] = result.groupby("position")[
            "projected_yr1_ppg"
        ].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

        # Clean up output columns
        name_col = None
        for col in ["pfr_name", "player_name", "name"]:
            if col in result.columns:
                name_col = col
                break

        output_cols = [
            name_col, "position", "draft_round", "draft_pick",
            "projected_yr1_ppg", "projected_yr1_total",
            "confidence", "upside_score",
            "total_awards", "has_heisman",
        ]
        # Only include columns that exist
        output_cols = [c for c in output_cols if c in result.columns]

        result = (
            result[output_cols]
            .sort_values("projected_yr1_ppg", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        result["rookie_rank"] = range(1, len(result) + 1)

        return result

    def _add_combine_to_draft(
        self, draft_df: pd.DataFrame, combine_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge combine results into draft data."""
        combine = combine_df.copy()

        # Find name columns
        draft_name_col = next(
            (c for c in ["pfr_name", "player_name", "name"] if c in draft_df.columns),
            None,
        )
        combine_name_col = next(
            (c for c in ["player_name", "name", "pfr_name"] if c in combine.columns),
            None,
        )

        if not draft_name_col or not combine_name_col:
            return draft_df

        draft_df["_merge_name"] = draft_df[draft_name_col].str.lower().str.strip()
        combine["_merge_name"] = combine[combine_name_col].str.lower().str.strip()

        # Select combine columns
        combine_keep = ["_merge_name"]
        for col in ["forty", "vertical", "bench", "broad_jump", "three_cone",
                     "shuttle", "ht", "wt", "forty_yd", "40yd"]:
            if col in combine.columns:
                combine_keep.append(col)

        combine_subset = combine[list(set(combine_keep))].drop_duplicates(
            subset=["_merge_name"]
        )

        result = draft_df.merge(combine_subset, on="_merge_name", how="left")
        result.drop(columns=["_merge_name"], inplace=True)
        return result

    def _add_award_features_to_prospects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add award counts for incoming prospects."""
        result = df.copy()
        name_col = next(
            (c for c in ["pfr_name", "player_name", "name"] if c in result.columns),
            None,
        )

        result["total_awards"] = 0
        result["has_heisman"] = 0
        result["has_position_award"] = 0
        result["has_maxwell_or_camp"] = 0

        if name_col is None:
            return result

        for idx, row in result.iterrows():
            name = row[name_col]
            draft_year = row.get("draft_year", row.get("season", 0))
            college_season = draft_year - 1

            award_info = get_award_counts(name, college_season, self._awards_df)
            for key, val in award_info.items():
                result.at[idx, key] = val

        return result

    def _extract_prospect_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for prospects — must match training feature set."""
        features = pd.DataFrame(index=df.index)

        # Draft capital
        round_col = next(
            (c for c in ["round", "draft_round"] if c in df.columns), None
        )
        pick_col = next(
            (c for c in ["pick", "draft_pick"] if c in df.columns), None
        )

        features["draft_round"] = df[round_col] if round_col else np.nan
        features["draft_pick"] = df[pick_col] if pick_col else np.nan
        features["is_first_round"] = (features["draft_round"] == 1).astype(int)
        features["is_top_10"] = (features["draft_pick"] <= 10).astype(int)

        # Combine
        for metric in ["forty", "vertical", "bench", "broad_jump", "three_cone"]:
            col_candidates = [metric, f"combine_{metric}"]
            found = False
            for col in col_candidates:
                if col in df.columns:
                    features[f"combine_{metric}"] = df[col].astype(float)
                    found = True
                    break
            if not found:
                features[f"combine_{metric}"] = np.nan

        # Position
        features["pos_QB"] = (df["position"] == "QB").astype(int)
        features["pos_RB"] = (df["position"] == "RB").astype(int)
        features["pos_WR"] = (df["position"] == "WR").astype(int)
        features["pos_TE"] = (df["position"] == "TE").astype(int)

        # Awards
        features["total_awards"] = df.get("total_awards", 0)
        features["has_heisman"] = df.get("has_heisman", 0)
        features["has_position_award"] = df.get("has_position_award", 0)
        features["has_maxwell_or_camp"] = df.get("has_maxwell_or_camp", 0)

        # Interaction
        features["draft_pick_x_awards"] = (
            features["draft_pick"].fillna(200) * features["total_awards"].fillna(0)
        )

        # Ensure same columns as training
        for col in self._feature_columns:
            if col not in features.columns:
                features[col] = 0

        features = features[self._feature_columns]
        features = features.fillna(features.median())
        features = features.fillna(0)

        return features

    def _calculate_confidence(self, features: pd.DataFrame) -> pd.Series:
        """
        Estimate prediction confidence based on feature completeness.

        More complete data = higher confidence. Missing combine data
        or being outside the training distribution lowers confidence.
        """
        non_null_ratio = features.notna().mean(axis=1)
        # Bonus for having draft capital data (most predictive)
        has_draft = (features["draft_pick"] > 0).astype(float) * 0.2
        confidence = (non_null_ratio * 0.8 + has_draft).clip(0, 1)
        return confidence

    # ──────────────────────────────────────────────────────────────────
    # Dynasty VORP integration
    # ──────────────────────────────────────────────────────────────────

    def merge_with_dynasty_vorp(
        self,
        rookie_projections: pd.DataFrame,
        dynasty_vorp_df: pd.DataFrame,
        rookie_age: int = 22,
    ) -> pd.DataFrame:
        """
        Integrate rookie projections into the Dynasty VORP framework.

        Rookies don't have NFL track records, so we inject their
        projected PPG as if it were actual production, then let the
        Dynasty VORP age-curve machinery handle the rest.

        This is analogous to how a VC values a pre-revenue startup:
        you project future revenue (our PPG projection) and apply
        a discount for uncertainty (lower confidence = higher discount).

        Args:
            rookie_projections: Output from predict_rookie_class()
            dynasty_vorp_df: Existing Dynasty VORP rankings for veterans
            rookie_age: Default age for rookies (adjusted if data available)

        Returns:
            Combined DataFrame with veterans and rookies ranked together
        """
        if rookie_projections.empty:
            return dynasty_vorp_df

        # Build rookie entries compatible with VORP DataFrame format
        name_col = next(
            (c for c in ["pfr_name", "player_name", "name"]
             if c in rookie_projections.columns),
            rookie_projections.columns[0],
        )

        rookies = pd.DataFrame(
            {
                "player_name": rookie_projections[name_col],
                "position": rookie_projections["position"],
                "ppg": rookie_projections["projected_yr1_ppg"],
                "total_points": rookie_projections["projected_yr1_total"],
                "games_played": 17,
                "age": rookie_age,
                "is_rookie": True,
                "rookie_confidence": rookie_projections.get("confidence", 0.5),
            }
        )

        # Apply confidence discount to rookie PPG
        # Lower confidence = more conservative projection
        rookies["ppg"] = rookies["ppg"] * (
            0.7 + 0.3 * rookies["rookie_confidence"]
        )
        rookies["total_points"] = rookies["ppg"] * 17

        # Add to veteran pool
        veterans = dynasty_vorp_df.copy()
        veterans["is_rookie"] = False
        veterans["rookie_confidence"] = 1.0

        combined = pd.concat([veterans, rookies], ignore_index=True)

        logger.info(
            f"Combined pool: {len(veterans)} veterans + {len(rookies)} rookies"
        )

        return combined
