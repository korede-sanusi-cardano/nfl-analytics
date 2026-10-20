"""
Tests for the redraft trade analyser.

All tests run offline: Sleeper and ESPN clients are replaced by small fakes.

Run with: pytest tests/test_redraft_trade.py -v
"""

import pytest

from src.data.espn_client import EspnPrivateLeagueError
from src.models.redraft_trade import (
    LeagueSnapshot,
    NFLPlayer,
    NFLWeekData,
    RedraftTradeAnalyser,
    ScoringRules,
    Team,
    load_espn_league,
    load_sleeper_league,
    normalise_name,
)

POSITIONS = ["QB", "RB", "RB", "RB", "WR", "WR", "WR", "TE"]
CURRENT_WEEK = 6


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


def _proj(points: float, position: str) -> dict:
    """Projection stats that score `points` in full PPR."""
    if position == "QB":
        return {"pass_yd": points / 0.04}
    return {"rec_yd": points / 0.1}


def _name(pid: str) -> str:
    """Letters-only test name (the name normaliser drops digits)."""
    return "Player " + "".join("abcdefghij"[int(d)] for d in pid)


def _make_player(pid: str, position: str, ppg: float, bye: int | None = None, stats_weeks: int = 0) -> NFLPlayer:
    p = NFLPlayer(pid, _name(pid), position, "AAA")
    for w in range(CURRENT_WEEK, 18):
        p.projections[w] = {"bye": w == bye, "stats": {} if w == bye else _proj(ppg, position)}
    for w in range(1, stats_weeks + 1):
        p.stats[w] = {**_proj(ppg, position), "rec_tgt": ppg / 2.0} if position != "QB" else {**_proj(ppg, position), "pass_att": ppg * 1.5}
    return p


@pytest.fixture
def nfl() -> NFLWeekData:
    players = {}
    n = 0
    # 4 teams x 8 rostered players, descending quality by team id; plus free agents
    for team in range(4):
        for pos in POSITIONS:
            n += 1
            ppg = 16 - 2 * team + (6 if pos == "QB" else 0)
            players[str(n)] = _make_player(str(n), pos, ppg, bye=8 + team, stats_weeks=5)
    for pos, ppg in (("RB", 9.0), ("WR", 9.5), ("TE", 7.0), ("QB", 14.0)):
        n += 1
        players[str(n)] = _make_player(str(n), pos, ppg)
    return NFLWeekData(season=2026, week=CURRENT_WEEK, players=players)


@pytest.fixture
def league(nfl) -> LeagueSnapshot:
    teams = {}
    for t in range(4):
        ids = [str(t * 8 + i + 1) for i in range(8)]
        teams[t + 1] = Team(t + 1, f"Team {t + 1}", f"Owner {t + 1}", wins=2, losses=3, points_for=600.0, players=ids)
    schedule = {w: [(1, 2), (3, 4)] if w % 2 else [(1, 3), (2, 4)] for w in range(CURRENT_WEEK, 15)}
    return LeagueSnapshot(
        platform="sleeper", league_id="test", name="Test League", teams=teams, schedule=schedule,
        starting_slots=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"], kd_slots=2, roster_size=10,
        playoff_teams=2, playoff_start=15, scoring=ScoringRules.full_ppr(), me=4,
    )


@pytest.fixture
def analyser(nfl, league) -> RedraftTradeAnalyser:
    return RedraftTradeAnalyser(nfl, league, market_weight=0.5)


# ══════════════════════════════════════════════════════════════════════
# Scoring & names
# ══════════════════════════════════════════════════════════════════════


class TestScoring:
    def test_full_ppr_score(self):
        s = ScoringRules.full_ppr()
        assert s.score({"rec": 5, "rec_yd": 60, "rec_td": 1}, "WR") == pytest.approx(17.0)
        assert s.score({"pass_yd": 250, "pass_td": 2, "pass_int": 1}, "QB") == pytest.approx(16.0)

    def test_sleeper_te_premium(self):
        s = ScoringRules.from_sleeper({"rec": 1, "rec_yd": 0.1, "bonus_rec_te": 0.5})
        assert s.score({"rec": 4}, "TE") == pytest.approx(6.0)
        assert s.score({"rec": 4}, "WR") == pytest.approx(4.0)

    def test_espn_scoring_items_and_overrides(self):
        items = [
            {"statId": 53, "points": 1.0, "pointsOverrides": {"6": 1.5}},
            {"statId": 42, "points": 0.1},
            {"statId": 4, "points": 4.0},
        ]
        s = ScoringRules.from_espn(items)
        assert s.weights["rec"] == 1.0
        assert s.rec_bonus["TE"] == pytest.approx(0.5)
        assert s.score({"rec": 2, "rec_yd": 20}, "TE") == pytest.approx(5.0)

    def test_espn_empty_scoring_defaults_to_ppr(self):
        assert ScoringRules.from_espn([]).weights["rec"] == 1

    @pytest.mark.parametrize(
        "a,b",
        [("Kenneth Walker III", "kenneth walker"), ("Ja'Marr Chase", "JaMarr Chase"), ("Travis Etienne Jr.", "Travis Etienne")],
    )
    def test_normalise_name(self, a, b):
        assert normalise_name(a) == normalise_name(b)


# ══════════════════════════════════════════════════════════════════════
# Valuation & lineups
# ══════════════════════════════════════════════════════════════════════


class TestValuation:
    def test_market_weight_one_returns_market(self, nfl, league):
        an = RedraftTradeAnalyser(nfl, league, market_weight=1.0)
        v = an.values["2"]  # team 1 RB, 16 ppg
        assert v.ppg == pytest.approx(v.market_ppg)
        assert v.market_ppg == pytest.approx(16.0)

    def test_bye_week_has_zero_mean(self, analyser):
        v = analyser.values["1"]  # team 1 bye week 8
        assert v.bye_week == 8
        assert v.weekly_mean[8] == 0 and v.weekly_availability[8] == 0

    def test_out_player_unavailable_this_week(self, nfl, league):
        nfl.players["2"].injury_status = "Out"
        an = RedraftTradeAnalyser(nfl, league)
        assert an.values["2"].weekly_availability[CURRENT_WEEK] == 0
        assert an.values["2"].weekly_availability[CURRENT_WEEK + 1] == pytest.approx(0.8)

    def test_no_future_projection_means_no_value(self, nfl, league):
        for w in nfl.players["2"].projections:
            nfl.players["2"].projections[w] = {"bye": False, "stats": {}}
        an = RedraftTradeAnalyser(nfl, league)
        assert an.values["2"].ppg == 0

    def test_lineup_fills_flex_with_best_remaining(self, analyser):
        roster = analyser.league.teams[1].players
        # QB 22 + 2 RB 16 + 2 WR 16 + TE 16 + FLEX (third RB/WR) 16, scaled by availability
        ev = analyser.lineup_ev(roster, CURRENT_WEEK)
        assert ev > 0
        assert ev == pytest.approx(analyser.lineup_ev([*roster, "33"], CURRENT_WEEK))  # FA bench adds nothing

    def test_better_roster_has_higher_ev(self, analyser):
        weeks = analyser.league.regular_weeks
        assert analyser.lineup_ev_avg(analyser.league.teams[1].players, weeks) > analyser.lineup_ev_avg(
            analyser.league.teams[4].players, weeks
        )


# ══════════════════════════════════════════════════════════════════════
# Simulation & trades
# ══════════════════════════════════════════════════════════════════════


class TestSimulation:
    def test_odds_sum_to_playoff_spots(self, analyser):
        res = analyser.simulate(n_sims=1000)
        assert sum(r["playoff_odds"] for r in res.values()) == pytest.approx(analyser.league.playoff_teams)
        assert sum(r["top_seed_odds"] for r in res.values()) == pytest.approx(1.0)

    def test_simulation_is_deterministic(self, analyser):
        a = analyser.simulate(n_sims=800)
        b = analyser.simulate(n_sims=800)
        assert a == b

    def test_stronger_team_more_likely(self, analyser):
        res = analyser.simulate(n_sims=1500)
        assert res[1]["playoff_odds"] > res[4]["playoff_odds"]

    def test_trade_for_better_player_raises_odds(self, analyser):
        # Team 4 swaps its RB (ppg 10) for team 1's RB (ppg 16)
        res = analyser.evaluate_trade(me=4, partner=1, give=["26"], get=["2"], n_sims=1500)
        assert res.playoff_odds_change > 0
        assert res.lineup_change > 0
        assert res.my_adds == []  # 1-for-1 opens no roster spot
        assert res.verdict() in ("accept", "accept_mutual")

    def test_two_for_one_fills_open_spot_from_free_agency(self, analyser):
        _, moves = analyser.apply_trade(me=4, partner=1, give=["26", "27"], get=["2"], auto_fill=True)
        assert len(moves["my_adds"]) == 1
        assert moves["my_adds"][0] in analyser.free_agents()

    def test_one_for_two_drops_lowest_marginal_player(self, analyser):
        analyser.league.roster_size = 8  # full rosters: receiving two for one forces a drop
        rosters, moves = analyser.apply_trade(me=1, partner=4, give=["2"], get=["26", "27"], auto_fill=False)
        assert len(rosters[1]) == 8
        assert len(moves["my_drops"]) == 1
        assert moves["my_drops"][0] not in ("26", "27")  # incoming players are protected

    def test_one_for_two_under_limit_keeps_everyone(self, analyser):
        rosters, moves = analyser.apply_trade(me=1, partner=4, give=["2"], get=["26", "27"], auto_fill=False)
        assert len(rosters[1]) == 9
        assert moves["my_drops"] == []

    def test_search_returns_mutual_improvements(self, analyser):
        results = analyser.search_trades(me=4, n_sims=600, shortlist=3)
        for r in results:
            assert r.lineup_change > 0 and r.partner_lineup_change > 0


# ══════════════════════════════════════════════════════════════════════
# Loaders (fake clients)
# ══════════════════════════════════════════════════════════════════════


class FakeSleeper:
    league_id = "L1"

    def get_league(self):
        return {
            "name": "Fake Sleeper",
            "settings": {"playoff_week_start": 15, "playoff_teams": 2},
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "K", "DEF", "BN", "BN", "IR"],
            "scoring_settings": {"rec": 1, "rec_yd": 0.1, "rush_yd": 0.1, "pass_yd": 0.04, "pass_td": 4},
        }

    def get_users(self):
        return [{"user_id": "u1", "display_name": "ksanusi17"}, {"user_id": "u2", "display_name": "rival"}]

    def get_rosters(self):
        return [
            {"roster_id": 1, "owner_id": "u1", "players": ["1", "2", "PIT"], "reserve": ["3"], "settings": {"wins": 3, "losses": 2, "fpts": 600, "fpts_decimal": 50}},
            {"roster_id": 2, "owner_id": "u2", "players": ["9", "10"], "reserve": [], "settings": {"wins": 2, "losses": 3, "fpts": 550}},
        ]

    def get_matchups(self, week):
        return [{"roster_id": 1, "matchup_id": 1}, {"roster_id": 2, "matchup_id": 1}]


class FakeEspn:
    league_id = "E1"

    def __init__(self, private=False):
        self.private = private

    def get_league(self):
        if self.private:
            raise EspnPrivateLeagueError("private")
        entry = lambda name, pos_id, slot: {"lineupSlotId": slot, "playerPoolEntry": {"player": {"fullName": name, "defaultPositionId": pos_id}}}  # noqa: E731
        return {
            "settings": {
                "name": "Fake ESPN",
                "scheduleSettings": {"matchupPeriodCount": 14, "playoffTeamCount": 2},
                "rosterSettings": {"lineupSlotCounts": {"0": 1, "2": 2, "4": 2, "6": 1, "23": 2, "16": 1, "17": 1, "20": 6, "21": 1}},
                "scoringSettings": {"scoringItems": [{"statId": 53, "points": 1.0}]},
            },
            "members": [{"id": "m1", "firstName": "Korede", "lastName": "Sanusi"}, {"id": "m2", "firstName": "A", "lastName": "B"}],
            "teams": [
                {"id": 3, "name": "Koredussy", "owners": ["m1"], "record": {"overall": {"wins": 3, "losses": 2, "pointsFor": 669.2}},
                 "roster": {"entries": [entry(_name("1"), 1, 0), entry(_name("2"), 2, 2), entry("Unknown Guy", 3, 4), entry("Broncos D/ST", 16, 16)]}},
                {"id": 6, "name": "Rival", "owners": ["m2"], "record": {"overall": {"wins": 3, "losses": 2, "pointsFor": 670.7}},
                 "roster": {"entries": [entry(_name("9"), 1, 0), entry(_name("10"), 2, 21)]}},
            ],
            "schedule": [
                {"matchupPeriodId": 5, "home": {"teamId": 3}, "away": {"teamId": 6}, "winner": "HOME"},
                {"matchupPeriodId": 6, "home": {"teamId": 3}, "away": {"teamId": 6}, "winner": "UNDECIDED"},
                {"matchupPeriodId": 15, "home": {"teamId": 3}, "away": {"teamId": 6}, "winner": "UNDECIDED"},
            ],
        }


class TestLoaders:
    def test_sleeper_loader(self, nfl):
        lg = load_sleeper_league(FakeSleeper(), nfl, me_username="KSANUSI17")
        assert lg.me == 1
        assert lg.teams[1].players == ["1", "2"]
        assert lg.teams[1].ir == ["3"]
        assert lg.teams[1].other_slots_used == 1  # PIT defence
        assert lg.teams[1].points_for == pytest.approx(600.5)
        assert "SFLEX" in lg.starting_slots and "FLEX" in lg.starting_slots
        assert lg.kd_slots == 2 and lg.roster_size == 12
        assert lg.regular_weeks == list(range(CURRENT_WEEK, 15))

    def test_espn_loader(self, nfl):
        lg = load_espn_league(FakeEspn(), nfl, me_first="korede", me_last="sanusi")
        assert lg.me == 3
        assert lg.teams[3].players == ["1", "2"]
        assert lg.teams[3].other_slots_used == 2  # D/ST + unmatched player
        assert lg.teams[6].ir == ["10"]
        assert lg.unmatched_players == ["Unknown Guy"]
        assert lg.schedule == {6: [(3, 6)]}  # decided and playoff weeks excluded
        assert lg.starting_slots.count("FLEX") == 2 and lg.kd_slots == 2

    def test_espn_private_league_raises(self, nfl):
        with pytest.raises(EspnPrivateLeagueError):
            load_espn_league(FakeEspn(private=True), nfl)
