"""
Sleeper API Client.

Wraps the Sleeper API (https://docs.sleeper.com) for fetching league,
roster, player, trade, and draft data. No authentication required.
"""

import time
from typing import Any

import pandas as pd
import requests
from loguru import logger


class SleeperClient:
    """Client for the Sleeper Fantasy Football API."""

    BASE_URL = "https://api.sleeper.app/v1"

    def __init__(self, league_id: str, rate_limit_per_min: int = 60):
        self.league_id = league_id
        self.min_interval = 60.0 / rate_limit_per_min
        self._last_request_time = 0.0
        self._player_cache: dict[str, Any] | None = None

    def _rate_limited_get(self, url: str) -> Any:
        """GET with rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        response = requests.get(url, timeout=30)
        self._last_request_time = time.time()
        response.raise_for_status()
        return response.json()

    # -- Core endpoints --

    def get_league(self) -> dict:
        """Get league settings and metadata."""
        return self._rate_limited_get(f"{self.BASE_URL}/league/{self.league_id}")

    def get_rosters(self) -> list[dict]:
        """Get all rosters in the league."""
        return self._rate_limited_get(
            f"{self.BASE_URL}/league/{self.league_id}/rosters"
        )

    def get_users(self) -> list[dict]:
        """Get all users in the league."""
        return self._rate_limited_get(
            f"{self.BASE_URL}/league/{self.league_id}/users"
        )

    def get_matchups(self, week: int) -> list[dict]:
        """Get matchups for a specific week."""
        return self._rate_limited_get(
            f"{self.BASE_URL}/league/{self.league_id}/matchups/{week}"
        )

    def get_transactions(self, week: int) -> list[dict]:
        """Get all transactions (trades, waivers, free agents) for a week."""
        return self._rate_limited_get(
            f"{self.BASE_URL}/league/{self.league_id}/transactions/{week}"
        )

    def get_traded_picks(self) -> list[dict]:
        """Get all traded draft picks."""
        return self._rate_limited_get(
            f"{self.BASE_URL}/league/{self.league_id}/traded_picks"
        )

    def get_drafts(self) -> list[dict]:
        """Get all drafts for the league."""
        return self._rate_limited_get(
            f"{self.BASE_URL}/league/{self.league_id}/drafts"
        )

    def get_draft_picks(self, draft_id: str) -> list[dict]:
        """Get all picks for a specific draft."""
        return self._rate_limited_get(
            f"{self.BASE_URL}/draft/{draft_id}/picks"
        )

    def get_all_players(self) -> dict[str, Any]:
        """Get all NFL players. Cached — this endpoint is large (~15MB)."""
        if self._player_cache is None:
            logger.info("Fetching full player database from Sleeper (this may take a moment)...")
            self._player_cache = self._rate_limited_get(
                f"{self.BASE_URL}/players/nfl"
            )
            logger.info(f"Cached {len(self._player_cache)} players")
        return self._player_cache

    def get_trending_players(
        self, sport: str = "nfl", trend_type: str = "add", lookback_hours: int = 24, limit: int = 25
    ) -> list[dict]:
        """Get trending adds or drops across the Sleeper platform."""
        return self._rate_limited_get(
            f"{self.BASE_URL}/players/{sport}/trending/{trend_type}"
            f"?lookback_hours={lookback_hours}&limit={limit}"
        )

    def get_user_by_username(self, username: str) -> dict:
        """Look up a Sleeper user by username."""
        return self._rate_limited_get(f"{self.BASE_URL}/user/{username}")

    def get_user_leagues(
        self, user_id: str, sport: str = "nfl", season: str = "2024"
    ) -> list[dict]:
        """Get all leagues for a user in a given season."""
        return self._rate_limited_get(
            f"{self.BASE_URL}/user/{user_id}/leagues/{sport}/{season}"
        )

    # -- Convenience methods --

    def get_league_history(self, current_league_id: str | None = None) -> list[str]:
        """Traverse league history to get all past season league IDs."""
        league_ids = []
        lid = current_league_id or self.league_id

        while lid:
            league_ids.append(lid)
            league_info = self._rate_limited_get(f"{self.BASE_URL}/league/{lid}")
            lid = league_info.get("previous_league_id")

        logger.info(f"Found {len(league_ids)} seasons of league history")
        return league_ids

    def get_all_trades(self) -> pd.DataFrame:
        """Get all trades across all weeks for the current season."""
        trades = []
        for week in range(1, 19):
            try:
                txns = self.get_transactions(week)
                week_trades = [t for t in txns if t.get("type") == "trade"]
                trades.extend(week_trades)
            except requests.HTTPError:
                break

        if not trades:
            return pd.DataFrame()

        return pd.DataFrame(trades)

    def build_roster_map(self) -> dict[int, str]:
        """Map roster_id to display name."""
        users = self.get_users()
        rosters = self.get_rosters()

        user_map = {u["user_id"]: u.get("display_name", u["user_id"]) for u in users}
        roster_map = {}
        for r in rosters:
            owner_id = r.get("owner_id", "")
            roster_map[r["roster_id"]] = user_map.get(owner_id, f"Unknown ({owner_id})")

        return roster_map
