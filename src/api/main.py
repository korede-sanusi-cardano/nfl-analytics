"""
NFL Dynasty Analytics — FastAPI Backend.

Serves VORP rankings, injury adjustments, draft state, and all
analytics models as REST endpoints for the React frontend.

Also acts as a proxy to the Sleeper API, solving CORS issues
that blocked direct browser → Sleeper calls.

Run with:
    uvicorn src.api.main:app --reload --port 8000

Endpoints:
    GET  /api/health                          — health check
    GET  /api/league/{league_id}              — league info
    GET  /api/league/{league_id}/rosters      — all rosters with user names
    GET  /api/rankings                        — VORP + Dynasty VORP rankings
    GET  /api/rankings/injured                — rankings with injury adjustments
    POST /api/injuries                        — add/update injury records
    GET  /api/injuries                        — current injury report
    GET  /api/injuries/types                  — list valid injury types
    POST /api/injuries/load-sleeper           — pull injuries from Sleeper
    GET  /api/scarcity                        — positional scarcity analysis
    GET  /api/aging-curves                    — aging curve data
    GET  /api/draft/{league_id}/drafts        — list drafts for a league
    GET  /api/draft/{league_id}/state/{id}    — live draft state + picks
    GET  /api/draft/{league_id}/available/{id}— best available by VORP
    GET  /api/sleeper/{path:path}             — proxy any Sleeper API call
"""

import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.sleeper_client import SleeperClient
from src.models.injury_model import INJURY_CATALOG, InjuryAdjuster
from src.models.vorp import (
    AGING_CURVES,
    LeagueSettings,
    VORPCalculator,
    get_retention_factor,
)


# ──────────────────────────────────────────────────────────────────────
# App state
# ──────────────────────────────────────────────────────────────────────

class AppState:
    """Shared state across requests — caches and model instances."""

    def __init__(self):
        self.injury_adjuster = InjuryAdjuster()
        self.sleeper_clients: dict[str, SleeperClient] = {}
        self.player_db_cache: dict[str, Any] | None = None

    def get_client(self, league_id: str) -> SleeperClient:
        if league_id not in self.sleeper_clients:
            self.sleeper_clients[league_id] = SleeperClient(league_id=league_id)
        return self.sleeper_clients[league_id]


state = AppState()

# Known injuries for the current cycle — update before each draft season
DEFAULT_INJURIES = [
    {
        "player_name": "Tyreek Hill",
        "position": "WR",
        "injury_type": "broken_leg",
        "severity": "major",
        "injury_date": "2024-11-24",
        "surgery": True,
        "notes": "Fractured femur/hip area vs Green Bay. Surgery performed.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.injury_adjuster.load_manual_overrides(DEFAULT_INJURIES)
    logger.info(
        f"API started. {state.injury_adjuster.injury_count} injuries pre-loaded."
    )
    yield


app = FastAPI(
    title="NFL Dynasty Analytics API",
    description="VORP rankings, injury adjustments, and live draft helper",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────

class InjuryInput(BaseModel):
    player_name: str
    position: str
    injury_type: str
    severity: str = "moderate"
    injury_date: str | None = None
    surgery: bool = False
    notes: str = ""


class InjuryBatchInput(BaseModel):
    injuries: list[InjuryInput]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _build_player_pool(
    league_id: str | None = None,
    season: int = 2024,
    positions: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a player pool with PPG and age data.

    Primary: nfl_data_py for real historical stats.
    Fallback: Sleeper player DB for names/positions/age (no PPG).
    """
    positions = positions or ["QB", "RB", "WR", "TE"]

    try:
        from src.data.nfl_stats import NFLStats

        nfl = NFLStats()
        frames = []
        for pos in positions:
            df = nfl.get_top_performers(season, pos, top_n=60)
            df["season"] = season
            frames.append(df)
        pool = pd.concat(frames, ignore_index=True)
        logger.info(f"Built player pool from nfl_data_py: {len(pool)} players")
        return pool

    except Exception as e:
        logger.warning(f"nfl_data_py unavailable ({e}), falling back to Sleeper")

    if not league_id:
        return pd.DataFrame()

    try:
        client = state.get_client(league_id)
        players = client.get_all_players()
        rows = []
        for pid, info in players.items():
            pos = info.get("position", "")
            if pos not in positions:
                continue
            name = info.get("full_name", "")
            if not name:
                continue
            rows.append({
                "player_id": pid,
                "player_name": name,
                "position": pos,
                "team": info.get("team", ""),
                "age": info.get("age"),
                "ppg": 0,
                "total_points": 0,
                "games_played": 0,
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"Sleeper fallback also failed: {e}")
        return pd.DataFrame()


def _ensure_pool_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["ppg", "total_points", "games_played", "age"]:
        if col not in df.columns:
            df[col] = 0
    return df


def _normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", name)
    name = re.sub(r"[.'`]", "", name)
    return " ".join(name.split())


# ──────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "injuries_tracked": state.injury_adjuster.injury_count,
        "injury_types_available": list(INJURY_CATALOG.keys()),
    }


# ──────────────────────────────────────────────────────────────────────
# Sleeper proxy (solves CORS for the frontend)
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/sleeper/{path:path}")
async def sleeper_proxy(path: str):
    url = f"https://api.sleeper.app/v1/{path}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Sleeper API error: {e}")


# ──────────────────────────────────────────────────────────────────────
# League info
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/league/{league_id}")
async def get_league(league_id: str):
    try:
        return state.get_client(league_id).get_league()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/league/{league_id}/rosters")
async def get_rosters(league_id: str):
    client = state.get_client(league_id)
    try:
        rosters = client.get_rosters()
        users = client.get_users()
        user_map = {
            u["user_id"]: u.get("display_name", u["user_id"]) for u in users
        }
        return [
            {
                "roster_id": r["roster_id"],
                "owner_name": user_map.get(r.get("owner_id", ""), "Unknown"),
                "owner_id": r.get("owner_id", ""),
                "players": r.get("players", []),
                "starters": r.get("starters", []),
                "wins": r.get("settings", {}).get("wins", 0),
                "losses": r.get("settings", {}).get("losses", 0),
                "fpts": r.get("settings", {}).get("fpts", 0),
            }
            for r in rosters
        ]
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ──────────────────────────────────────────────────────────────────────
# VORP Rankings
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/rankings")
async def get_rankings(
    league_id: str = Query(default="1311316930342195200"),
    num_teams: int = Query(default=12),
    scoring: str = Query(default="ppr"),
    projection_years: int = Query(default=5),
    discount_rate: float = Query(default=0.10),
    top_n: int = Query(default=100),
    season: int = Query(default=2024),
    positions: str = Query(default="QB,RB,WR,TE"),
):
    """Dynasty VORP rankings (no injury adjustments)."""
    pos_list = [p.strip() for p in positions.split(",")]
    pool = _build_player_pool(league_id, season, pos_list)
    if pool.empty:
        raise HTTPException(status_code=404, detail="No player data available")

    pool = _ensure_pool_columns(pool)
    settings = LeagueSettings(num_teams=num_teams, scoring=scoring)
    calc = VORPCalculator(settings)
    dynasty_df = calc.calculate_dynasty_vorp(
        pool,
        projection_years=projection_years,
        discount_rate=discount_rate,
    )
    return dynasty_df.head(top_n).to_dict("records")


@app.get("/api/rankings/injured")
async def get_rankings_with_injuries(
    league_id: str = Query(default="1311316930342195200"),
    num_teams: int = Query(default=12),
    projection_years: int = Query(default=5),
    discount_rate: float = Query(default=0.10),
    top_n: int = Query(default=100),
    season: int = Query(default=2024),
    positions: str = Query(default="QB,RB,WR,TE"),
):
    """Dynasty VORP rankings WITH injury adjustments applied."""
    pos_list = [p.strip() for p in positions.split(",")]
    pool = _build_player_pool(league_id, season, pos_list)
    if pool.empty:
        raise HTTPException(status_code=404, detail="No player data available")

    pool = _ensure_pool_columns(pool)
    settings = LeagueSettings(num_teams=num_teams)
    calc = VORPCalculator(settings)
    dynasty_df = calc.calculate_dynasty_vorp(
        pool,
        projection_years=projection_years,
        discount_rate=discount_rate,
    )

    adjusted = state.injury_adjuster.adjust_dynasty_vorp(
        dynasty_df,
        projection_years=projection_years,
        discount_rate=discount_rate,
    )
    return adjusted.head(top_n).to_dict("records")


# ──────────────────────────────────────────────────────────────────────
# Injuries
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/injuries")
async def get_injuries():
    report = state.injury_adjuster.injury_report()
    return report.to_dict("records") if not report.empty else []


@app.post("/api/injuries")
async def add_injury(injury: InjuryInput):
    if injury.injury_type not in INJURY_CATALOG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown injury type '{injury.injury_type}'. "
            f"Valid: {list(INJURY_CATALOG.keys())}",
        )
    state.injury_adjuster.add_injury(
        player_name=injury.player_name,
        position=injury.position,
        injury_type=injury.injury_type,
        severity=injury.severity,
        injury_date=injury.injury_date,
        surgery=injury.surgery,
        notes=injury.notes,
        source="api",
    )
    return {"status": "ok", "injuries_tracked": state.injury_adjuster.injury_count}


@app.post("/api/injuries/batch")
async def add_injuries_batch(batch: InjuryBatchInput):
    count = 0
    for inj in batch.injuries:
        if inj.injury_type not in INJURY_CATALOG:
            continue
        state.injury_adjuster.add_injury(
            player_name=inj.player_name,
            position=inj.position,
            injury_type=inj.injury_type,
            severity=inj.severity,
            injury_date=inj.injury_date,
            surgery=inj.surgery,
            notes=inj.notes,
            source="api",
        )
        count += 1
    return {"status": "ok", "added": count, "total": state.injury_adjuster.injury_count}


@app.get("/api/injuries/types")
async def get_injury_types():
    return {
        key: {
            "label": val["label"],
            "recovery_seasons": val["recovery_seasons"],
            "severities": list(val["games_missed"].keys()),
        }
        for key, val in INJURY_CATALOG.items()
    }


@app.post("/api/injuries/load-sleeper")
async def load_sleeper_injuries(
    league_id: str = Query(default="1311316930342195200"),
):
    client = state.get_client(league_id)
    count = state.injury_adjuster.load_sleeper_injuries(client)
    return {"status": "ok", "injuries_loaded": count}


# ──────────────────────────────────────────────────────────────────────
# Scarcity
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/scarcity")
async def get_scarcity(
    league_id: str = Query(default="1311316930342195200"),
    num_teams: int = Query(default=12),
    season: int = Query(default=2024),
):
    pool = _build_player_pool(league_id, season)
    if pool.empty:
        raise HTTPException(status_code=404, detail="No player data available")

    pool = _ensure_pool_columns(pool)
    settings = LeagueSettings(num_teams=num_teams)
    calc = VORPCalculator(settings)
    scarcity = calc.positional_scarcity(pool)
    return scarcity.to_dict("records") if not scarcity.empty else []


# ──────────────────────────────────────────────────────────────────────
# Aging curves
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/aging-curves")
async def get_aging_curves(
    source: str = Query(default="model", description="'model' for VORP curves, 'empirical' for nfl_data_py"),
    seasons_start: int = Query(default=2018),
    seasons_end: int = Query(default=2024),
):
    """
    Return aging curve data.

    source='model' → returns the retention factors from vorp.py
    source='empirical' → pulls real data from nfl_data_py
    """
    if source == "empirical":
        try:
            from src.data.nfl_stats import NFLStats

            nfl = NFLStats()
            seasons = list(range(seasons_start, seasons_end + 1))
            result = {}
            for pos in ["QB", "RB", "WR", "TE"]:
                curve = nfl.get_aging_curves(seasons, pos)
                result[pos] = curve.to_dict("records")
            return result
        except Exception as e:
            logger.warning(f"Empirical curves unavailable: {e}. Falling back to model.")

    # Model-based curves
    data = []
    for age in range(20, 40):
        point = {"age": age}
        for pos in ["QB", "RB", "WR", "TE"]:
            point[f"{pos}_retention"] = round(get_retention_factor(pos, age), 3)
        data.append(point)
    return data


# ──────────────────────────────────────────────────────────────────────
# Live Draft
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/draft/{league_id}/drafts")
async def get_drafts(league_id: str):
    """List all drafts for a league."""
    try:
        return state.get_client(league_id).get_drafts()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/draft/{league_id}/state/{draft_id}")
async def get_draft_state(league_id: str, draft_id: str):
    """
    Get full live draft state: metadata + picks + derived info.

    This is the primary endpoint the React draft board polls.
    Returns everything the frontend needs in a single call.
    """
    client = state.get_client(league_id)
    try:
        draft_info = client.get_draft(draft_id)
        picks = client.get_draft_picks(draft_id)
        users = client.get_users()

        user_map = {
            u["user_id"]: u.get("display_name", u["user_id"]) for u in users
        }

        settings = draft_info.get("settings", {})
        total_rounds = int(settings.get("rounds", 15))
        total_teams = int(settings.get("teams", 12))
        total_picks = total_rounds * total_teams
        picks_made = len(picks)
        current_pick_no = picks_made + 1
        current_round = (picks_made // total_teams) + 1
        pick_in_round = (picks_made % total_teams) + 1

        # Enrich picks with readable names
        enriched_picks = []
        for p in picks:
            meta = p.get("metadata", {})
            enriched_picks.append({
                "pick_no": p.get("pick_no"),
                "round": p.get("round"),
                "draft_slot": p.get("draft_slot"),
                "player_id": p.get("player_id"),
                "player_name": f"{meta.get('first_name', '')} {meta.get('last_name', '')}".strip(),
                "position": meta.get("position", ""),
                "team": meta.get("team", ""),
                "picked_by": user_map.get(
                    p.get("picked_by", ""), f"Slot {p.get('draft_slot', '?')}"
                ),
            })

        return {
            "draft_id": draft_id,
            "status": draft_info.get("status", "unknown"),
            "season": draft_info.get("season"),
            "type": draft_info.get("type"),
            "total_rounds": total_rounds,
            "total_teams": total_teams,
            "total_picks": total_picks,
            "picks_made": picks_made,
            "current_pick_no": current_pick_no,
            "current_round": min(current_round, total_rounds),
            "pick_in_round": pick_in_round,
            "picks": enriched_picks,
            "slot_to_user": draft_info.get("slot_to_roster_id", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/draft/{league_id}/available/{draft_id}")
async def get_draft_available(
    league_id: str,
    draft_id: str,
    my_slot: int = Query(default=1, description="Your draft slot (1-based)"),
    num_teams: int = Query(default=12),
    projection_years: int = Query(default=5),
    discount_rate: float = Query(default=0.10),
    top_n: int = Query(default=40),
    season: int = Query(default=2024),
    include_injuries: bool = Query(default=True),
    positions: str = Query(default="QB,RB,WR,TE"),
):
    """
    Best available players ranked by Dynasty VORP, with drafted
    players removed. This is the core of the live draft helper.

    Also returns positional need for the requesting user's slot.
    """
    client = state.get_client(league_id)
    pos_list = [p.strip() for p in positions.split(",")]

    try:
        picks = client.get_draft_picks(draft_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Build set of drafted player names (normalised)
    drafted_names: set[str] = set()
    my_picks: list[dict] = []

    for p in picks:
        meta = p.get("metadata", {})
        full = f"{meta.get('first_name', '')} {meta.get('last_name', '')}".strip()
        if full:
            drafted_names.add(_normalize_name(full))
        if str(p.get("draft_slot")) == str(my_slot):
            my_picks.append({
                "round": p.get("round"),
                "pick_no": p.get("pick_no"),
                "player_name": full,
                "position": meta.get("position", ""),
            })

    # Build VORP rankings
    pool = _build_player_pool(league_id, season, pos_list)
    if pool.empty:
        raise HTTPException(status_code=404, detail="No player data available")

    pool = _ensure_pool_columns(pool)
    settings = LeagueSettings(num_teams=num_teams)
    calc = VORPCalculator(settings)
    dynasty_df = calc.calculate_dynasty_vorp(
        pool,
        projection_years=projection_years,
        discount_rate=discount_rate,
    )

    # Apply injury adjustments if requested
    if include_injuries:
        dynasty_df = state.injury_adjuster.adjust_dynasty_vorp(
            dynasty_df,
            projection_years=projection_years,
            discount_rate=discount_rate,
        )
        rank_col = "injury_adjusted_vorp"
    else:
        rank_col = "dynasty_vorp"

    # Remove drafted players
    available = dynasty_df[
        ~dynasty_df["player_name"].apply(_normalize_name).isin(drafted_names)
    ].copy()
    available = available.sort_values(rank_col, ascending=False).head(top_n)

    # Positional need for my slot
    starter_targets = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    pos_counts = {}
    for p in my_picks:
        pos = p["position"]
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    positional_need = {
        pos: max(0, target - pos_counts.get(pos, 0))
        for pos, target in starter_targets.items()
    }

    return {
        "available": available.to_dict("records"),
        "my_picks": my_picks,
        "positional_need": positional_need,
        "drafted_count": len(drafted_names),
        "rank_column": rank_col,
    }


# ──────────────────────────────────────────────────────────────────────
# Drop-off predictions (if model is trained)
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/dropoff")
async def get_dropoff_predictions(
    season: int = Query(default=2024),
    top_n: int = Query(default=50),
):
    """
    Get drop-off risk predictions for current players.

    Requires the model to be trained first — returns 503 if not.
    """
    try:
        from src.models.dropoff_predictor import DropoffPredictor
        from src.data.nfl_stats import NFLStats

        nfl = NFLStats()
        predictor = DropoffPredictor(nfl)

        # Train on historical data
        metrics = predictor.train()
        if "error" in metrics:
            raise HTTPException(status_code=503, detail=metrics["error"])

        # Get current player data
        pool = _build_player_pool(season=season)
        if pool.empty:
            raise HTTPException(status_code=404, detail="No player data")

        pool = _ensure_pool_columns(pool)
        result = predictor.predict_dropoff(pool.head(top_n))

        return {
            "predictions": result.to_dict("records"),
            "model_metrics": {
                "cv_auc": metrics["cv_auc_mean"],
                "training_samples": metrics["training_samples"],
                "dropoff_rate": metrics["dropoff_rate"],
            },
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Drop-off model not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────
# Rookie projections (if model is trained)
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/rookies")
async def get_rookie_projections(
    draft_year: int = Query(default=2025),
    top_n: int = Query(default=30),
):
    """
    Get rookie fantasy projections for an incoming draft class.

    Requires training data via nfl_data_py — returns 503 if unavailable.
    """
    try:
        from src.models.rookie_model import RookieValuationModel
        from src.data.nfl_stats import NFLStats

        nfl = NFLStats()
        model = RookieValuationModel(nfl)
        metrics = model.train()

        if "error" in metrics:
            raise HTTPException(status_code=503, detail=metrics["error"])

        projections = model.predict_rookie_class(draft_year, top_n=top_n)
        if projections.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No draft data for {draft_year}",
            )

        return {
            "projections": projections.to_dict("records"),
            "model_metrics": {
                "cv_mae": metrics["cv_mae_mean"],
                "training_samples": metrics["training_samples"],
            },
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Rookie model not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
