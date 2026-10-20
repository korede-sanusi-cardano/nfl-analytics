"""
ESPN Fantasy Football API Client.

Wraps the read-only ESPN fantasy API used by fantasy.espn.com. Public
leagues need no authentication. Private leagues return HTTP 401 unless
the league manager enables "Make league viewable to public"; optional
``espn_s2`` / ``swid`` cookies can be supplied for private leagues when
calling from a server (never from a browser).
"""

from typing import Any, ClassVar

import requests
from loguru import logger


class EspnPrivateLeagueError(RuntimeError):
    """Raised when ESPN refuses a league because it is private."""


class EspnClient:
    """Client for the ESPN Fantasy Football (ffl) league API."""

    BASE_URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons"

    # Lineup slot ids used by ESPN
    SLOT_NAMES: ClassVar[dict[int, str]] = {
        0: "QB",
        2: "RB",
        4: "WR",
        6: "TE",
        23: "FLEX",
        7: "SFLEX",
        3: "WRRB_FLEX",
        5: "REC_FLEX",
        16: "DEF",
        17: "K",
        20: "BN",
        21: "IR",
    }
    POSITION_IDS: ClassVar[dict[int, str]] = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DEF"}

    # ESPN statId -> Sleeper-style stat key (only those that score in standard leagues)
    STAT_KEYS: ClassVar[dict[int, str]] = {
        3: "pass_yd",
        4: "pass_td",
        20: "pass_int",
        19: "pass_2pt",
        24: "rush_yd",
        25: "rush_td",
        26: "rush_2pt",
        53: "rec",
        42: "rec_yd",
        43: "rec_td",
        44: "rec_2pt",
        72: "fum_lost",
    }

    def __init__(
        self,
        league_id: str,
        season: int,
        espn_s2: str | None = None,
        swid: str | None = None,
        timeout: int = 30,
    ):
        self.league_id = league_id
        self.season = season
        self.timeout = timeout
        self._cookies = {"espn_s2": espn_s2, "SWID": swid} if espn_s2 and swid else None

    def _get(self, views: list[str]) -> dict[str, Any]:
        url = f"{self.BASE_URL}/{self.season}/segments/0/leagues/{self.league_id}"
        params = [("view", v) for v in views]
        response = requests.get(url, params=params, cookies=self._cookies, timeout=self.timeout)
        if response.status_code in (401, 403):
            raise EspnPrivateLeagueError(
                f"ESPN league {self.league_id} is private. Ask the league manager to enable "
                "'Make league viewable to public' (League > Settings > Basic Settings)."
            )
        response.raise_for_status()
        return response.json()

    def get_league(self) -> dict[str, Any]:
        """Get teams, members, rosters, settings and the full schedule in one call."""
        logger.info(f"Fetching ESPN league {self.league_id} ({self.season})")
        return self._get(["mTeam", "mRoster", "mSettings", "mMatchupScore"])
