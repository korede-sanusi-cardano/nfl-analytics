"""
Redraft Trade Analyser.

Values an in-season redraft trade by the change it makes to each side's
chance of reaching the fantasy playoffs, not by summing player points.

Pipeline
--------
1. **Player value (rest of season).** Two views are blended:

   * *Market*: Sleeper's week-by-week projections (already include byes,
     opponents and short-term injuries).
   * *Usage*: this season's opportunity — carries, targets, air yards,
     red-zone looks — converted to expected fantasy points with
     coefficients fitted on 2025-26 nflverse play-by-play (R^2 ~= 0.97),
     plus a heavily shrunk efficiency term so touchdown luck mostly washes
     out. The market view acts as a Bayesian prior worth ``PRIOR_GAMES``.

   Think of it as a credibility-weighted estimate: the market number is the
   consensus price, this season's usage is our own signal, and the weight
   on our signal grows with the number of games observed.

2. **Lineup expected value.** For each remaining week, fill the league's
   starting slots greedily (dedicated slots first, then flex slots from most
   to least restrictive).

3. **Monte Carlo.** Each remaining regular-season week is simulated for every
   team: availability is drawn first (so a manager can start a bench player),
   then scores from a gamma distribution whose spread grows with the
   projection. Seeding uses wins, then points for. Before and after a trade
   are run on the same random draws per player-week (common random numbers),
   so the difference is a paired comparison with low noise.

The module is platform-agnostic: ``LeagueSnapshot`` is built from either a
Sleeper or an ESPN league by the loader functions at the bottom.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
import zlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from src.data.espn_client import EspnClient
from src.data.sleeper_client import SleeperClient

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

SKILL_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

# Stat keys that carry fantasy points in standard scoring
SCORING_KEYS: tuple[str, ...] = (
    "pass_yd", "pass_td", "pass_int", "pass_2pt",
    "rush_yd", "rush_td", "rush_2pt",
    "rec", "rec_yd", "rec_td", "rec_2pt",
    "fum_lost",
)
# Usage keys needed for the opportunity model
USAGE_KEYS: tuple[str, ...] = (
    "rush_att", "rush_rz_att", "rec_tgt", "rec_air_yd", "rec_rz_tgt",
    "pass_att", "pass_rz_att", "gp", "off_snp",
)

# Usage -> expected full-PPR points per game (fitted on nflverse pbp 2025-26)
SKILL_XFP_COEF: dict[str, float] = {
    "rush_att": 0.4867, "rush_rz_att": 0.908, "rec_tgt": 1.3562,
    "rec_air_yd": 0.0247, "rec_rz_tgt": 1.0727,
}
QB_XFP_COEF: dict[str, float] = {
    "pass_att": 0.3215, "pass_rz_att": 0.7574, "rush_att": 0.5166, "rush_rz_att": 0.8459,
}
CATCH_RATE = 0.65  # converts target-based xFP when reception value != 1 point

# Weekly score spread: sd = a + b * mean (fitted on 2025 weekly PPR scores)
WEEKLY_SD: dict[str, tuple[float, float]] = {
    "QB": (7.0, 0.062), "RB": (4.11, 0.25), "WR": (3.59, 0.295), "TE": (2.16, 0.426),
}
# Shrinkage on efficiency (actual - expected): games of "zero" added per position
EFFICIENCY_SHRINK: dict[str, float] = {"QB": 8, "RB": 12, "WR": 10, "TE": 12}
PRIOR_GAMES = 3.0
BASE_AVAILABILITY: dict[str, float] = {"QB": 0.96, "RB": 0.92, "WR": 0.94, "TE": 0.94}
KD_MEAN_PER_SLOT = 8.5  # kicker / defence treated as league-average
KD_SD_PER_SLOT = 4.25

SLOT_ELIGIBILITY: dict[str, tuple[str, ...]] = {
    "QB": ("QB",), "RB": ("RB",), "WR": ("WR",), "TE": ("TE",),
    "FLEX": ("RB", "WR", "TE"), "SFLEX": ("QB", "RB", "WR", "TE"),
    "WRRB_FLEX": ("RB", "WR"), "REC_FLEX": ("WR", "TE"),
}


def normalise_name(name: str) -> str:
    """Lower-case, strip accents, suffixes (Jr, II...) and punctuation."""
    name = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", name.lower())
    return re.sub(r"[^a-z]", "", name)


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ScoringRules:
    """Points per stat, plus position-specific reception bonuses (e.g. TE premium)."""

    weights: dict[str, float]
    rec_bonus: dict[str, float] = field(default_factory=dict)

    @classmethod
    def full_ppr(cls) -> ScoringRules:
        return cls(
            weights={
                "pass_yd": 0.04, "pass_td": 4, "pass_int": -2, "pass_2pt": 2,
                "rush_yd": 0.1, "rush_td": 6, "rush_2pt": 2,
                "rec": 1, "rec_yd": 0.1, "rec_td": 6, "rec_2pt": 2, "fum_lost": -2,
            }
        )

    @classmethod
    def from_sleeper(cls, scoring_settings: dict[str, float]) -> ScoringRules:
        weights = {k: float(scoring_settings.get(k, 0) or 0) for k in SCORING_KEYS}
        rec_bonus = {
            "RB": float(scoring_settings.get("bonus_rec_rb", 0) or 0),
            "WR": float(scoring_settings.get("bonus_rec_wr", 0) or 0),
            "TE": float(scoring_settings.get("bonus_rec_te", 0) or 0),
        }
        return cls(weights=weights, rec_bonus=rec_bonus)

    @classmethod
    def from_espn(cls, scoring_items: list[dict]) -> ScoringRules:
        if not scoring_items:
            return cls.full_ppr()
        weights = dict.fromkeys(SCORING_KEYS, 0.0)
        rec_bonus: dict[str, float] = {}
        for item in scoring_items:
            key = EspnClient.STAT_KEYS.get(item.get("statId"))
            if not key:
                continue
            weights[key] = float(item.get("points", 0))
            overrides = item.get("pointsOverrides") or {}
            if key == "rec":
                for slot_id, pos in ((2, "RB"), (4, "WR"), (6, "TE")):
                    if str(slot_id) in overrides:
                        rec_bonus[pos] = float(overrides[str(slot_id)]) - weights[key]
        return cls(weights=weights, rec_bonus=rec_bonus)

    def score(self, stats: dict[str, float] | None, position: str) -> float:
        if not stats:
            return 0.0
        pts = sum(stats.get(k, 0) * w for k, w in self.weights.items() if stats.get(k))
        pts += stats.get("rec", 0) * self.rec_bonus.get(position, 0.0)
        return float(pts)

    def reception_value(self, position: str) -> float:
        return self.weights.get("rec", 0.0) + self.rec_bonus.get(position, 0.0)


@dataclass
class NFLPlayer:
    """One NFL skill player with weekly projections and this season's stats."""

    player_id: str  # Sleeper id
    name: str
    position: str
    team: str | None
    injury_status: str | None = None
    projections: dict[int, dict[str, Any]] = field(default_factory=dict)  # week -> {bye, stats}
    stats: dict[int, dict[str, float]] = field(default_factory=dict)  # week -> stats


@dataclass
class NFLWeekData:
    """League-independent NFL data for the season: projections + box scores."""

    season: int
    week: int
    players: dict[str, NFLPlayer]

    def by_name(self) -> dict[str, str]:
        """Map normalised 'name|POS' -> Sleeper player id (prefers players with a team)."""
        out: dict[str, str] = {}
        for p in self.players.values():
            key = f"{normalise_name(p.name)}|{p.position}"
            if key not in out or (p.team and not self.players[out[key]].team):
                out[key] = p.player_id
        return out


@dataclass
class Team:
    team_id: int
    name: str
    owner: str
    wins: float
    losses: int
    points_for: float
    players: list[str]  # active skill players (Sleeper ids), IR excluded
    ir: list[str] = field(default_factory=list)
    other_slots_used: int = 0  # K/DEF and unmatched players occupying roster spots


@dataclass
class LeagueSnapshot:
    """Everything needed to value trades in one league, platform-agnostic."""

    platform: str
    league_id: str
    name: str
    teams: dict[int, Team]
    schedule: dict[int, list[tuple[int, int]]]  # remaining regular-season week -> pairs
    starting_slots: list[str]
    kd_slots: int
    roster_size: int  # non-IR roster spots
    playoff_teams: int
    playoff_start: int
    scoring: ScoringRules
    me: int | None = None
    unmatched_players: list[str] = field(default_factory=list)

    @property
    def regular_weeks(self) -> list[int]:
        return sorted(self.schedule)

    @property
    def playoff_weeks(self) -> list[int]:
        return list(range(self.playoff_start, min(17, self.playoff_start + 2) + 1))


@dataclass
class PlayerValue:
    player_id: str
    name: str
    position: str
    team: str | None
    injury_status: str | None
    ppg: float  # blended rest-of-season points per game
    market_ppg: float
    usage_ppg: float
    games: int
    bye_week: int | None
    weekly_mean: dict[int, float]
    weekly_availability: dict[int, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "position": self.position,
            "team": self.team,
            "injury_status": self.injury_status,
            "ppg": round(self.ppg, 2),
            "market_ppg": round(self.market_ppg, 2),
            "usage_ppg": round(self.usage_ppg, 2),
            "games": self.games,
            "bye_week": self.bye_week,
        }


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────


def _pick(stats: dict | None, keys: Iterable[str]) -> dict[str, float]:
    if not stats:
        return {}
    return {k: float(stats[k]) for k in keys if stats.get(k)}


def load_nfl_week_data(client: SleeperClient, last_week: int = 17, workers: int = 6) -> NFLWeekData:
    """Fetch live Sleeper projections (current week -> ``last_week``) and box-score
    stats (week 1 -> last completed week)."""
    state = client.get_nfl_state()
    season = int(state.get("league_season") or state.get("season"))
    week = int(state.get("week") or state.get("display_week") or 1)
    players: dict[str, NFLPlayer] = {}

    def touch(entry: dict) -> NFLPlayer | None:
        info = entry.get("player") or {}
        pos = info.get("position")
        if pos not in SKILL_POSITIONS:
            return None
        pid = str(entry["player_id"])
        p = players.get(pid)
        if p is None:
            name = f"{info.get('first_name', '')} {info.get('last_name', '')}".strip()
            p = NFLPlayer(pid, name, pos, entry.get("team") or info.get("team"))
            players[pid] = p
        if entry.get("team"):
            p.team = entry["team"]
        if "injury_status" in info:
            p.injury_status = info.get("injury_status")
        return p

    proj_weeks = list(range(week, last_week + 1))
    stat_weeks = list(range(1, week))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        proj = dict(zip(proj_weeks, pool.map(lambda w: client.get_weekly_projections(season, w), proj_weeks), strict=True))
        stat = dict(zip(stat_weeks, pool.map(lambda w: client.get_weekly_player_stats(season, w), stat_weeks), strict=True))

    for w, entries in proj.items():
        for e in entries or []:
            p = touch(e)
            if p:
                p.projections[w] = {"bye": not e.get("opponent"), "stats": _pick(e.get("stats"), SCORING_KEYS)}
    for w, entries in stat.items():
        for e in entries or []:
            p = touch(e)
            if not p:
                continue
            s = e.get("stats") or {}
            played = (s.get("gp") or 0) > 0 or (s.get("off_snp") or 0) > 0 or (
                (s.get("rush_att") or 0) + (s.get("rec_tgt") or 0) + (s.get("pass_att") or 0)
            ) > 0
            if played:
                p.stats[w] = {**_pick(s, SCORING_KEYS), **_pick(s, USAGE_KEYS)}

    logger.info(f"Loaded {len(players)} NFL players: projections wk{week}-{last_week}, stats wk1-{week - 1}")
    return NFLWeekData(season=season, week=week, players=players)


def load_sleeper_league(client: SleeperClient, nfl: NFLWeekData, me_username: str | None = None) -> LeagueSnapshot:
    """Build a LeagueSnapshot for a Sleeper league (client.league_id)."""
    league = client.get_league()
    users = client.get_users()
    rosters = client.get_rosters()
    settings = league.get("settings", {})
    playoff_start = int(settings.get("playoff_week_start") or 15)
    weeks = list(range(nfl.week, playoff_start))
    schedule: dict[int, list[tuple[int, int]]] = {}
    for w in weeks:
        groups: dict[int, list[int]] = {}
        for m in client.get_matchups(w) or []:
            if m.get("matchup_id") is not None:
                groups.setdefault(m["matchup_id"], []).append(m["roster_id"])
        schedule[w] = [(g[0], g[1]) for g in groups.values() if len(g) == 2]

    user_map = {u["user_id"]: u for u in users}
    teams: dict[int, Team] = {}
    me = None
    for r in rosters:
        u = user_map.get(r.get("owner_id"), {})
        s = r.get("settings") or {}
        reserve = r.get("reserve") or []
        active = [p for p in (r.get("players") or []) if p not in reserve]
        skill = [p for p in active if p in nfl.players]
        teams[r["roster_id"]] = Team(
            team_id=r["roster_id"],
            name=(u.get("metadata") or {}).get("team_name") or u.get("display_name") or f"Team {r['roster_id']}",
            owner=u.get("display_name", ""),
            wins=(s.get("wins") or 0) + 0.5 * (s.get("ties") or 0),
            losses=s.get("losses") or 0,
            points_for=(s.get("fpts") or 0) + (s.get("fpts_decimal") or 0) / 100,
            players=skill,
            ir=[p for p in reserve if p in nfl.players],
            other_slots_used=len(active) - len(skill),
        )
        if me_username and (u.get("display_name") or "").lower() == me_username.lower():
            me = r["roster_id"]

    positions = league.get("roster_positions") or []
    starting = [("SFLEX" if s == "SUPER_FLEX" else s) for s in positions if s in SLOT_ELIGIBILITY or s == "SUPER_FLEX"]
    return LeagueSnapshot(
        platform="sleeper",
        league_id=str(client.league_id),
        name=league.get("name", "Sleeper league"),
        teams=teams,
        schedule=schedule,
        starting_slots=starting,
        kd_slots=sum(1 for s in positions if s in ("K", "DEF")),
        roster_size=sum(1 for s in positions if s not in ("IR", "TAXI")),
        playoff_teams=int(settings.get("playoff_teams") or 6),
        playoff_start=playoff_start,
        scoring=ScoringRules.from_sleeper(league.get("scoring_settings") or {}),
        me=me,
    )


def load_espn_league(
    client: EspnClient,
    nfl: NFLWeekData,
    me_first: str | None = None,
    me_last: str | None = None,
) -> LeagueSnapshot:
    """Build a LeagueSnapshot for an ESPN league, matching players to Sleeper ids by name."""
    data = client.get_league()
    st = data.get("settings") or {}
    sched_settings = st.get("scheduleSettings") or {}
    slot_counts = (st.get("rosterSettings") or {}).get("lineupSlotCounts") or {}
    starting: list[str] = []
    kd = 0
    size = 0
    for sid, count in slot_counts.items():
        n, slot = int(count or 0), EspnClient.SLOT_NAMES.get(int(sid))
        if not n or slot is None:
            continue
        if slot != "IR":
            size += n
        if slot in ("K", "DEF"):
            kd += n
        elif slot in SLOT_ELIGIBILITY:
            starting += [slot] * n

    last_reg = int(sched_settings.get("matchupPeriodCount") or 14)
    lookup = nfl.by_name()
    members = {m["id"]: m for m in data.get("members") or []}
    teams: dict[int, Team] = {}
    unmatched: list[str] = []
    me = None
    for t in data.get("teams") or []:
        rec = (t.get("record") or {}).get("overall") or {}
        owners = [members[o] for o in t.get("owners") or [] if o in members]
        players, ir, other = [], [], 0
        for e in (t.get("roster") or {}).get("entries") or []:
            pl = (e.get("playerPoolEntry") or {}).get("player") or {}
            pos = EspnClient.POSITION_IDS.get(pl.get("defaultPositionId"))
            on_ir = e.get("lineupSlotId") == 21
            if pos not in SKILL_POSITIONS:
                other += 0 if on_ir else 1
                continue
            sid = lookup.get(f"{normalise_name(pl.get('fullName', ''))}|{pos}")
            if sid is None:
                unmatched.append(pl.get("fullName", "?"))
                other += 0 if on_ir else 1
                continue
            (ir if on_ir else players).append(sid)
        teams[t["id"]] = Team(
            team_id=t["id"],
            name=t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip() or f"Team {t['id']}",
            owner=", ".join(f"{o.get('firstName', '')} {o.get('lastName', '')}".strip() for o in owners),
            wins=(rec.get("wins") or 0) + 0.5 * (rec.get("ties") or 0),
            losses=rec.get("losses") or 0,
            points_for=float(rec.get("pointsFor") or 0),
            players=players,
            ir=ir,
            other_slots_used=other,
        )
        if me_first and me_last and any(
            normalise_name(o.get("firstName")) == normalise_name(me_first)
            and normalise_name(o.get("lastName")) == normalise_name(me_last)
            for o in owners
        ):
            me = t["id"]

    schedule: dict[int, list[tuple[int, int]]] = {}
    for m in data.get("schedule") or []:
        w = m.get("matchupPeriodId")
        if w is None or w > last_reg or w < nfl.week or not m.get("home") or not m.get("away"):
            continue
        if m.get("winner") not in (None, "UNDECIDED"):
            continue
        schedule.setdefault(w, []).append((m["home"]["teamId"], m["away"]["teamId"]))

    if unmatched:
        logger.warning(f"{len(unmatched)} ESPN players not matched to Sleeper ids: {unmatched[:10]}")
    return LeagueSnapshot(
        platform="espn",
        league_id=str(client.league_id),
        name=st.get("name") or f"ESPN {client.league_id}",
        teams=teams,
        schedule=schedule,
        starting_slots=starting,
        kd_slots=kd,
        roster_size=size,
        playoff_teams=int(sched_settings.get("playoffTeamCount") or 6),
        playoff_start=last_reg + 1,
        scoring=ScoringRules.from_espn((st.get("scoringSettings") or {}).get("scoringItems") or []),
        me=me,
        unmatched_players=unmatched,
    )


# ──────────────────────────────────────────────────────────────────────
# The analyser
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TradeResult:
    me: int
    partner: int
    give: list[str]
    get: list[str]
    playoff_odds_before: float
    playoff_odds_after: float
    partner_odds_before: float
    partner_odds_after: float
    expected_wins_change: float
    lineup_change: float  # my expected starting lineup, pts/wk, rest of regular season
    playoff_week_lineup_change: float
    partner_lineup_change: float
    my_drops: list[str]
    my_adds: list[str]
    partner_drops: list[str]
    sims: int

    @property
    def playoff_odds_change(self) -> float:
        return self.playoff_odds_after - self.playoff_odds_before

    def verdict(self) -> str:
        near_lock = self.playoff_odds_before > 0.85
        score = self.playoff_odds_change + (self.playoff_week_lineup_change / 100 if near_lock else 0)
        if score >= 0.02 and self.partner_lineup_change >= 0:
            return "accept_mutual"  # good for you, also improves their lineup
        if score >= 0.02:
            return "accept"
        if score > -0.02:
            return "neutral"
        return "decline"


class RedraftTradeAnalyser:
    """Value redraft trades in one league by change in playoff probability."""

    def __init__(self, nfl: NFLWeekData, league: LeagueSnapshot, market_weight: float = 0.5):
        self.nfl = nfl
        self.league = league
        self.market_weight = float(np.clip(market_weight, 0, 1))
        self.values: dict[str, PlayerValue] = {}
        self._draw_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._free_agents: list[str] | None = None
        self._value_players()

    # -- valuation -------------------------------------------------------

    def _usage_xfp(self, stats: dict[str, float], position: str) -> float:
        if position == "QB":
            return sum(c * stats.get(k, 0) for k, c in QB_XFP_COEF.items())
        base = sum(c * stats.get(k, 0) for k, c in SKILL_XFP_COEF.items())
        return base + (self.league.scoring.reception_value(position) - 1) * CATCH_RATE * stats.get("rec_tgt", 0)

    def value_player(self, p: NFLPlayer) -> PlayerValue:
        cur = self.nfl.week
        last = max(max(self.league.regular_weeks, default=cur), self.league.playoff_weeks[-1])
        weekly_proj = {}
        for w in range(cur, last + 1):
            pr = p.projections.get(w)
            weekly_proj[w] = 0.0 if not pr or pr["bye"] else self.league.scoring.score(pr["stats"], p.position)
        live = [v for v in weekly_proj.values() if v > 0.5]
        market = float(np.mean(live)) if live else 0.0

        games = len(p.stats)
        usage = market
        if games and market > 0:
            xs = sum(self._usage_xfp(s, p.position) for s in p.stats.values())
            eff = sum(self.league.scoring.score(s, p.position) - self._usage_xfp(s, p.position) for s in p.stats.values())
            usage = (xs + PRIOR_GAMES * market) / (games + PRIOR_GAMES) + eff / (games + EFFICIENCY_SHRINK[p.position])
        if market == 0:
            usage = 0.0  # no projected games left: out for season / released
        ppg = self.market_weight * market + (1 - self.market_weight) * usage

        inj = (p.injury_status or "").lower()
        mean, avail = {}, {}
        for w, pw in weekly_proj.items():
            if pw <= 0.5:
                mean[w], avail[w] = 0.0, 0.0
                continue
            mean[w] = ppg * float(np.clip(pw / market, 0.7, 1.3))
            a = BASE_AVAILABILITY[p.position]
            if w == cur:
                a = {"questionable": 0.75, "doubtful": 0.3, "out": 0.0}.get(inj, a)
            elif w == cur + 1 and inj in ("doubtful", "out"):
                a = 0.8
            avail[w] = a
        bye = next((w for w, pr in sorted(p.projections.items()) if pr["bye"]), None)
        return PlayerValue(p.player_id, p.name, p.position, p.team, p.injury_status, ppg, market, usage,
                           games, bye, mean, avail)

    def _value_players(self) -> None:
        owned = {s for t in self.league.teams.values() for s in (*t.players, *t.ir)}
        for p in self.nfl.players.values():
            if p.team or p.player_id in owned:
                self.values[p.player_id] = self.value_player(p)

    # -- lineups ---------------------------------------------------------

    def _slot_plan(self) -> list[str]:
        return sorted(self.league.starting_slots, key=lambda s: len(SLOT_ELIGIBILITY[s]))

    def lineup_ev(self, roster: Iterable[str], week: int) -> float:
        """Expected points from the best starting lineup in one week (skill slots only)."""
        vals = sorted(
            ((v.position, v.weekly_mean.get(week, 0) * v.weekly_availability.get(week, 0))
             for s in roster if (v := self.values.get(s))),
            key=lambda x: -x[1],
        )
        slots = self._slot_plan()
        used = [False] * len(slots)
        total = 0.0
        for pos, e in vals:
            if e <= 0:
                break
            for i, slot in enumerate(slots):
                if not used[i] and pos in SLOT_ELIGIBILITY[slot]:
                    used[i] = True
                    total += e
                    break
        return total

    def lineup_ev_avg(self, roster: Iterable[str], weeks: list[int]) -> float:
        roster = list(roster)
        return float(np.mean([self.lineup_ev(roster, w) for w in weeks])) if weeks else 0.0

    # -- simulation ------------------------------------------------------

    def _draws(self, sid: str, week: int, n: int) -> tuple[float, np.ndarray, np.ndarray]:
        v = self.values.get(sid)
        mu = v.weekly_mean.get(week, 0.0) if v else 0.0
        a = v.weekly_availability.get(week, 0.0) if v else 0.0
        key = (sid, week, n, round(mu, 4), a)
        if key not in self._draw_cache:
            rng = np.random.default_rng([zlib.crc32(sid.encode()), week, zlib.crc32(self.league.league_id.encode())])
            active = rng.random(n) < a
            if mu > 0:
                a_, b_ = WEEKLY_SD[v.position]
                sd = a_ + b_ * mu
                pts = rng.gamma((mu / sd) ** 2, sd * sd / mu, n)
            else:
                pts = np.zeros(n)
            self._draw_cache[key] = (active, pts)
        active, pts = self._draw_cache[key]
        return mu, active, pts

    def _team_week(self, team_id: int, roster: list[str], week: int, n: int) -> np.ndarray:
        slots = self._slot_plan()
        players = []
        for s in roster:
            if s not in self.values:
                continue
            mu, active, pts = self._draws(s, week, n)
            if mu > 0:
                players.append((mu, self.values[s].position, active, pts))
        players.sort(key=lambda x: -x[0])
        open_slots = np.ones((n, len(slots)), dtype=bool)
        rows = np.arange(n)
        total = np.zeros(n)
        for _, pos, active, pts in players:
            elig = np.array([pos in SLOT_ELIGIBILITY[s] for s in slots])
            cand = open_slots & elig & active[:, None]
            has = cand.any(axis=1)
            first = cand.argmax(axis=1)
            open_slots[rows[has], first[has]] = False
            total += np.where(has, pts, 0.0)
        kd = self.league.kd_slots
        rng = np.random.default_rng([team_id, week, 999, zlib.crc32(self.league.league_id.encode())])
        total += rng.normal(KD_MEAN_PER_SLOT * kd, KD_SD_PER_SLOT * np.sqrt(max(kd, 1)), n)
        return total

    def simulate(self, rosters: dict[int, list[str]] | None = None, n_sims: int = 5000) -> dict[int, dict[str, float]]:
        """Simulate the rest of the regular season. Returns per team: playoff odds,
        expected wins and odds of the top seed."""
        rosters = rosters or {t: list(team.players) for t, team in self.league.teams.items()}
        ids = sorted(self.league.teams)
        idx = {t: i for i, t in enumerate(ids)}
        wins = np.array([[self.league.teams[t].wins] * n_sims for t in ids], dtype=float)
        pf = np.array([[self.league.teams[t].points_for] * n_sims for t in ids], dtype=float)
        for w in self.league.regular_weeks:
            scores = {t: self._team_week(t, rosters[t], w, n_sims) for t in ids}
            for a, b in self.league.schedule[w]:
                if a not in idx or b not in idx:
                    continue
                sa, sb = scores[a], scores[b]
                wins[idx[a]] += (sa > sb) + 0.5 * (sa == sb)
                wins[idx[b]] += (sb > sa) + 0.5 * (sa == sb)
                pf[idx[a]] += sa
                pf[idx[b]] += sb
        rank = (-(wins * 1e6 + pf)).argsort(axis=0).argsort(axis=0)
        return {
            t: {
                "playoff_odds": float((rank[i] < self.league.playoff_teams).mean()),
                "expected_wins": float(wins[i].mean()),
                "top_seed_odds": float((rank[i] == 0).mean()),
            }
            for t, i in idx.items()
        }

    # -- trades ----------------------------------------------------------

    def free_agents(self, limit: int = 30) -> list[str]:
        if self._free_agents is None:
            owned = {s for t in self.league.teams.values() for s in (*t.players, *t.ir)}
            fa = [v for s, v in self.values.items() if s not in owned and v.team and v.ppg > 3]
            self._free_agents = [v.player_id for v in sorted(fa, key=lambda v: -v.ppg)[:limit]]
        return list(self._free_agents)

    def apply_trade(
        self, me: int, partner: int, give: list[str], get: list[str], auto_fill: bool = True
    ) -> tuple[dict[int, list[str]], dict[str, list[str]]]:
        """Swap players, then drop the lowest-marginal player where over the roster
        limit and (optionally) add the best free agent where under it."""
        rosters = {t: list(team.players) for t, team in self.league.teams.items()}
        my_size_before = len(rosters[me])
        rosters[me] = [s for s in rosters[me] if s not in give] + list(get)
        rosters[partner] = [s for s in rosters[partner] if s not in get] + list(give)
        weeks = self.league.regular_weeks
        moves: dict[str, list[str]] = {"my_drops": [], "my_adds": [], "partner_drops": []}
        fa = self.free_agents()

        def fix(team_id: int, protect: list[str], drop_key: str, fill: bool) -> None:
            roster = rosters[team_id]
            limit = self.league.roster_size - self.league.teams[team_id].other_slots_used
            while len(roster) > limit:
                base = self.lineup_ev_avg(roster, weeks)
                cands = [s for s in roster if s not in protect]
                if not cands:
                    break
                worst = min(cands, key=lambda s: base - self.lineup_ev_avg([x for x in roster if x != s], weeks))
                roster.remove(worst)
                moves[drop_key].append(worst)
            # only refill spots the trade opened, never spots that were already empty
            target = min(limit, my_size_before)
            while fill and len(roster) < target and fa:
                base = self.lineup_ev_avg(roster, weeks)
                best = max((s for s in fa if s not in roster), key=lambda s: self.lineup_ev_avg([*roster, s], weeks), default=None)
                if best is None:
                    break
                roster.append(best)
                fa.remove(best)
                moves["my_adds"].append(best)
            rosters[team_id] = roster

        fix(me, list(get), "my_drops", auto_fill)
        fix(partner, list(give), "partner_drops", False)
        return rosters, moves

    def evaluate_trade(
        self,
        me: int,
        partner: int,
        give: list[str],
        get: list[str],
        n_sims: int = 5000,
        auto_fill: bool = True,
        baseline: dict[int, dict[str, float]] | None = None,
    ) -> TradeResult:
        baseline = baseline or self.simulate(n_sims=n_sims)
        rosters, moves = self.apply_trade(me, partner, give, get, auto_fill)
        after = self.simulate(rosters, n_sims)
        before_rosters = {t: list(team.players) for t, team in self.league.teams.items()}
        reg, po = self.league.regular_weeks, self.league.playoff_weeks
        return TradeResult(
            me=me, partner=partner, give=list(give), get=list(get),
            playoff_odds_before=baseline[me]["playoff_odds"],
            playoff_odds_after=after[me]["playoff_odds"],
            partner_odds_before=baseline[partner]["playoff_odds"],
            partner_odds_after=after[partner]["playoff_odds"],
            expected_wins_change=after[me]["expected_wins"] - baseline[me]["expected_wins"],
            lineup_change=self.lineup_ev_avg(rosters[me], reg) - self.lineup_ev_avg(before_rosters[me], reg),
            playoff_week_lineup_change=self.lineup_ev_avg(rosters[me], po) - self.lineup_ev_avg(before_rosters[me], po),
            partner_lineup_change=self.lineup_ev_avg(rosters[partner], reg) - self.lineup_ev_avg(before_rosters[partner], reg),
            my_drops=moves["my_drops"], my_adds=moves["my_adds"], partner_drops=moves["partner_drops"],
            sims=n_sims,
        )

    def search_trades(
        self,
        me: int,
        min_ppg: float = 7.0,
        min_gain: float = 0.3,
        shortlist: int = 10,
        n_sims: int = 3000,
        auto_fill: bool = True,
    ) -> list[TradeResult]:
        """Screen 1-for-1, 2-for-1 and 1-for-2 deals where both expected lineups
        improve, then rank the shortlist by change in playoff odds."""
        weeks = self.league.regular_weeks
        base = {t: list(team.players) for t, team in self.league.teams.items()}
        mine = [s for s in base[me] if s in self.values and self.values[s].ppg >= min_ppg]
        b0 = self.lineup_ev_avg(base[me], weeks)
        cands = []
        for t in self.league.teams:
            if t == me:
                continue
            theirs = [s for s in base[t] if s in self.values and self.values[s].ppg >= min_ppg]
            t0 = self.lineup_ev_avg(base[t], weeks)
            for gi, ge in ((1, 1), (2, 1), (1, 2)):
                for give in itertools.combinations(mine, gi):
                    for get in itertools.combinations(theirs, ge):
                        my_new = [s for s in base[me] if s not in give] + list(get)
                        th_new = [s for s in base[t] if s not in get] + list(give)
                        dm = self.lineup_ev_avg(my_new, weeks) - b0
                        dt = self.lineup_ev_avg(th_new, weeks) - t0
                        if dm > min_gain and dt > min_gain:
                            cands.append((dm, t, list(give), list(get)))
        cands.sort(key=lambda c: -c[0])
        baseline = self.simulate(n_sims=n_sims)
        results = [
            self.evaluate_trade(me, t, give, get, n_sims, auto_fill, baseline)
            for _, t, give, get in cands[: shortlist + 4]
        ]
        results = [r for r in results if r.lineup_change > 0 and r.partner_lineup_change > 0]
        return sorted(results, key=lambda r: -r.playoff_odds_change)[:shortlist]
