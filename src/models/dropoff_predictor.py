"""
Drop-off Predictor.

ML model to predict season-over-season performance decline.
Think of it as a credit risk model, but instead of predicting default
probability, you're predicting the probability a player's production
drops by >20% next season.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.data.nfl_stats import NFLStats


class DropoffPredictor:
    """Predict player production drop-offs using historical patterns."""

    DROPOFF_THRESHOLD = -0.20  # 20% decline = "drop-off"

    def __init__(self, nfl_stats: NFLStats):
        self.nfl_stats = nfl_stats
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self._is_fitted = False

    def build_training_data(
        self, seasons: list[int], min_games: int = 8
    ) -> pd.DataFrame:
        """
        Build feature set for drop-off prediction.

        Features mirror the kind of signals a credit analyst would use:
        - Age (like years since founding)
        - Usage trends (like revenue growth)
        - Efficiency metrics (like margins)
        - Workload (like leverage ratio)
        """
        weekly = self.nfl_stats.get_weekly_data(seasons)
        rosters = self.nfl_stats.get_rosters(seasons)

        # Season-level aggregation
        season_stats = (
            weekly.groupby(["player_id", "player_name", "season"])
            .agg(
                total_points=("fantasy_points_ppr", "sum"),
                games=("fantasy_points_ppr", "count"),
                avg_points=("fantasy_points_ppr", "mean"),
                std_points=("fantasy_points_ppr", "std"),
            )
            .reset_index()
        )
        season_stats = season_stats[season_stats["games"] >= min_games]

        # Add demographics
        roster_info = rosters.drop_duplicates(
            subset=["player_id", "season"]
        )[["player_id", "season", "position", "age"]].copy()

        merged = season_stats.merge(
            roster_info, on=["player_id", "season"], how="inner"
        )

        # Calculate YoY change
        merged = merged.sort_values(["player_id", "season"])
        merged["prev_ppg"] = merged.groupby("player_id")["avg_points"].shift(1)
        merged["ppg_change"] = (
            (merged["avg_points"] - merged["prev_ppg"]) / merged["prev_ppg"]
        )
        merged["dropped_off"] = (merged["ppg_change"] < self.DROPOFF_THRESHOLD).astype(int)

        # Only keep rows where we have prior season data
        merged = merged.dropna(subset=["prev_ppg", "ppg_change"])

        return merged

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract ML features from the training data."""
        features = pd.DataFrame()
        features["age"] = df["age"]
        features["prev_ppg"] = df["prev_ppg"]
        features["games_played"] = df["games"]
        features["consistency"] = df["std_points"] / df["avg_points"].clip(lower=0.1)
        features["position_QB"] = (df["position"] == "QB").astype(int)
        features["position_RB"] = (df["position"] == "RB").astype(int)
        features["position_WR"] = (df["position"] == "WR").astype(int)
        features["position_TE"] = (df["position"] == "TE").astype(int)

        return features.fillna(0)

    def train(self, seasons: list[int] | None = None) -> dict:
        """Train the drop-off prediction model."""
        if seasons is None:
            seasons = list(range(2015, 2024))

        logger.info(f"Building training data from {seasons[0]}-{seasons[-1]}")
        training_data = self.build_training_data(seasons)

        X = self.extract_features(training_data)
        y = training_data["dropped_off"]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Cross-validate
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring="roc_auc")
        logger.info(f"Cross-validation AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        # Fit on full data
        self.model.fit(X_scaled, y)
        self._is_fitted = True

        # Feature importance
        importance = dict(zip(X.columns, self.model.feature_importances_))

        return {
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "feature_importance": importance,
            "training_samples": len(X),
            "dropoff_rate": y.mean(),
        }

    def predict_dropoff(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict drop-off probability for current players.

        Returns the original DataFrame with added columns:
        - dropoff_probability: 0-1 probability of >20% decline
        - risk_tier: Low/Medium/High/Very High
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call .train() first.")

        X = self.extract_features(player_data)
        X_scaled = self.scaler.transform(X)

        probs = self.model.predict_proba(X_scaled)[:, 1]

        result = player_data.copy()
        result["dropoff_probability"] = probs
        result["risk_tier"] = pd.cut(
            probs,
            bins=[0, 0.25, 0.45, 0.65, 1.0],
            labels=["Low", "Medium", "High", "Very High"],
        )

        return result.sort_values("dropoff_probability", ascending=False)
