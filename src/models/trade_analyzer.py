"""
Trade Sentiment Analyzer.

Analyses trade history from your Sleeper league to understand how
specific managers value players. This is your information edge —
like having proprietary flow data in financial markets.
"""

from collections import defaultdict

import pandas as pd
from loguru import logger

from src.data.sleeper_client import SleeperClient


class TradeSentimentAnalyzer:
    """Analyse trade patterns to find exploitable biases."""

    def __init__(self, client: SleeperClient):
        self.client = client

    def get_trade_history(self) -> pd.DataFrame:
        """Pull all trades across the league's full history."""
        league_ids = self.client.get_league_history()
        all_trades = []

        for lid in league_ids:
            # Temporarily switch league ID
            original_id = self.client.league_id
            self.client.league_id = lid

            try:
                league_info = self.client.get_league()
                season = league_info.get("season", "unknown")

                for week in range(1, 19):
                    try:
                        txns = self.client.get_transactions(week)
                        trades = [t for t in txns if t.get("type") == "trade"]
                        for t in trades:
                            t["season"] = season
                            t["week"] = week
                        all_trades.extend(trades)
                    except Exception:
                        break
            finally:
                self.client.league_id = original_id

        logger.info(f"Found {len(all_trades)} total trades across {len(league_ids)} seasons")
        return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    def manager_tendencies(self, trades_df: pd.DataFrame) -> dict:
        """
        Analyse each manager's trading behaviour.

        Returns patterns like: who trades most, who hoards picks,
        who sells aging players, who buys high.
        """
        if trades_df.empty:
            return {}

        roster_map = self.client.build_roster_map()
        tendencies = defaultdict(
            lambda: {
                "total_trades": 0,
                "players_acquired": [],
                "players_traded_away": [],
                "picks_acquired": 0,
                "picks_traded_away": 0,
            }
        )

        for _, trade in trades_df.iterrows():
            adds = trade.get("adds") or {}
            drops = trade.get("drops") or {}
            draft_picks = trade.get("draft_picks") or []

            for player_id, roster_id in adds.items():
                manager = roster_map.get(roster_id, f"Roster {roster_id}")
                tendencies[manager]["total_trades"] += 1
                tendencies[manager]["players_acquired"].append(player_id)

            for player_id, roster_id in drops.items():
                manager = roster_map.get(roster_id, f"Roster {roster_id}")
                tendencies[manager]["players_traded_away"].append(player_id)

            for pick in draft_picks:
                owner = roster_map.get(pick.get("owner_id"), f"Roster {pick.get('owner_id')}")
                prev_owner = roster_map.get(
                    pick.get("previous_owner_id"),
                    f"Roster {pick.get('previous_owner_id')}",
                )
                tendencies[owner]["picks_acquired"] += 1
                tendencies[prev_owner]["picks_traded_away"] += 1

        return dict(tendencies)

    def cross_platform_analysis(self, username: str, season: str = "2025") -> pd.DataFrame:
        """
        Find a user's other leagues and compare their roster decisions.

        This is the real information edge: if your league mate dropped
        or traded a player in another league, they might be losing
        confidence in them — even if they haven't moved them in your league yet.
        """
        user_info = self.client.get_user_by_username(username)
        user_id = user_info["user_id"]
        leagues = self.client.get_user_leagues(user_id, season=season)

        logger.info(f"User {username} is in {len(leagues)} leagues in {season}")

        cross_platform_data = []
        for league in leagues:
            lid = league["league_id"]
            cross_platform_data.append(
                {
                    "league_id": lid,
                    "league_name": league.get("name", "Unknown"),
                    "total_rosters": league.get("total_rosters", 0),
                    "scoring": league.get("scoring_settings", {}),
                }
            )

        return pd.DataFrame(cross_platform_data)
