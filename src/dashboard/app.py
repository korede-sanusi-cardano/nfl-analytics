import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.nfl_stats import NFLStats
from src.data.sleeper_client import SleeperClient
from src.models.vorp import LeagueSettings, VORPCalculator


"""
NFL Dynasty Analytics — Streamlit Draft Helper Dashboard.

Your command centre for dynasty draft preparation. Displays VORP rankings,
aging curves, positional scarcity, and trade sentiment in one place.
"""

# Add project root to path

st.set_page_config(
    page_title="Dynasty Draft Helper",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def load_player_data(seasons: list[int]) -> pd.DataFrame:
    """Load and cache player performance data."""
    nfl = NFLStats()
    all_data = []

    for season in seasons:
        for pos in ["QB", "RB", "WR", "TE"]:
            df = nfl.get_top_performers(season, pos, top_n=60)
            df["season"] = season
            all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


@st.cache_data(ttl=3600)
def load_aging_curves(seasons: list[int]) -> dict[str, pd.DataFrame]:
    """Load aging curves for all positions."""
    nfl = NFLStats()
    curves = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        curves[pos] = nfl.get_aging_curves(seasons, pos)
    return curves


def main():
    st.title("🏈 Dynasty Draft Helper")
    st.markdown("*VORP-based rankings with dynasty age adjustments*")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")

        analysis_season = st.selectbox(
            "Analysis Season",
            options=list(range(2025, 2014, -1)),
            index=0,
        )

        projection_years = st.slider(
            "Dynasty Projection Window (years)",
            min_value=1,
            max_value=10,
            value=5,
            help="How many years to project for Dynasty VORP (like your DCF horizon)",
        )

        discount_rate = (
            st.slider(
                "Discount Rate (%)",
                min_value=5,
                max_value=25,
                value=10,
                help="Annual discount rate for future production (higher = more weight on present)",
            )
            / 100.0
        )

        num_teams = st.number_input("League Size", value=12, min_value=8, max_value=16)

        scoring_format = st.selectbox("Scoring", ["ppr", "half_ppr", "standard"])

        st.divider()
        st.subheader("Live Draft")
        sleeper_league_id = st.text_input(
            "Sleeper League ID",
            value="",
            help="Found in your Sleeper URL: sleeper.app/leagues/<league_id>",
        )
        my_pick_slot = st.number_input(
            "My Draft Pick Slot",
            min_value=1,
            max_value=16,
            value=1,
            help="Your pick position in the draft order (1 = first overall pick).",
        )
        auto_refresh_enabled = st.toggle("Auto-refresh", value=False)
        refresh_interval_secs = st.slider(
            "Refresh interval (s)",
            min_value=10,
            max_value=120,
            value=30,
            disabled=not auto_refresh_enabled,
        )

    # Load data
    with st.spinner("Loading player data..."):
        player_data = load_player_data([analysis_season])

    # Calculate VORP
    settings = LeagueSettings(
        num_teams=num_teams,
        scoring=scoring_format,
    )
    calc = VORPCalculator(settings)
    dynasty_df = calc.calculate_dynasty_vorp(
        player_data,
        projection_years=projection_years,
        discount_rate=discount_rate,
    )

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Dynasty Rankings",
            "📈 Aging Curves",
            "⚖️ Positional Scarcity",
            "🔍 Player Comparison",
            "🏈 Live Draft",
        ]
    )

    with tab1:
        st.subheader("Dynasty VORP Rankings")

        col1, col2 = st.columns([3, 1])
        with col2:
            pos_filter = st.multiselect(
                "Filter Position",
                ["QB", "RB", "WR", "TE"],
                default=["QB", "RB", "WR", "TE"],
            )

        filtered = dynasty_df[dynasty_df["position"].isin(pos_filter)].head(50)

        # Rankings table
        display_cols = [
            "dynasty_rank",
            "player_name",
            "position",
            "age",
            "ppg",
            "vorp",
            "dynasty_vorp",
            "games_played",
        ]
        st.dataframe(
            filtered[display_cols].style.format(
                {
                    "ppg": "{:.1f}",
                    "vorp": "{:.1f}",
                    "dynasty_vorp": "{:.1f}",
                }
            ),
            use_container_width=True,
            height=600,
        )

        # VORP distribution chart
        fig = px.scatter(
            filtered,
            x="age",
            y="dynasty_vorp",
            color="position",
            size="ppg",
            hover_name="player_name",
            title="Dynasty VORP by Age",
            labels={"dynasty_vorp": "Dynasty VORP", "age": "Age"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Empirical Aging Curves")
        st.markdown("*Based on historical NFL data — use to validate the model's assumptions*")

        history_range = st.slider(
            "Historical Data Range",
            min_value=2015,
            max_value=2025,
            value=(2018, 2025),
        )

        with st.spinner("Calculating aging curves..."):
            curves = load_aging_curves(list(range(history_range[0], history_range[1] + 1)))

        fig = go.Figure()
        for pos, curve_df in curves.items():
            fig.add_trace(
                go.Scatter(
                    x=curve_df["age"],
                    y=curve_df["avg_ppg"],
                    name=pos,
                    mode="lines+markers",
                )
            )

        fig.update_layout(
            title="Average PPG by Age and Position",
            xaxis_title="Age",
            yaxis_title="Points Per Game (PPR)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Positional Scarcity Analysis")
        st.markdown("*Higher scarcity = draft that position earlier*")

        scarcity = calc.positional_scarcity(player_data)
        st.dataframe(
            scarcity.style.format(
                {
                    "top5_vorp_share": "{:.1%}",
                    "top12_vorp_share": "{:.1%}",
                    "max_vorp": "{:.1f}",
                    "median_starter_vorp": "{:.1f}",
                    "dropoff": "{:.1f}",
                    "scarcity_score": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        fig = px.bar(
            scarcity,
            x="position",
            y="scarcity_score",
            color="position",
            title="Positional Scarcity Score",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Player Comparison Tool")

        col1, col2 = st.columns(2)
        player_names = dynasty_df["player_name"].tolist()

        with col1:
            player_a = st.selectbox("Player A", player_names, index=0)
        with col2:
            player_b = st.selectbox(
                "Player B",
                player_names,
                index=min(1, len(player_names) - 1),
            )

        if player_a and player_b:
            comp = dynasty_df[dynasty_df["player_name"].isin([player_a, player_b])]

            metrics = ["ppg", "vorp", "dynasty_vorp", "age", "games_played"]
            comp_display = comp[["player_name", "position"] + metrics].set_index("player_name")
            st.dataframe(
                comp_display.style.format(
                    {
                        "ppg": "{:.1f}",
                        "vorp": "{:.1f}",
                        "dynasty_vorp": "{:.1f}",
                    }
                ),
                use_container_width=True,
            )


    with tab5:
        _render_live_draft_tab(dynasty_df, sleeper_league_id, my_pick_slot,
                               auto_refresh_enabled, refresh_interval_secs)


def _normalize_name(name: str) -> str:
    """Lowercase, strip name suffixes, and collapse whitespace for fuzzy matching."""
    name = name.lower().strip()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", name)
    name = re.sub(r"[.'`]", "", name)
    return " ".join(name.split())


@st.cache_data(ttl=20)
def _fetch_draft_state(league_id: str, draft_id: str) -> tuple[dict, list[dict]]:
    """Fetch current draft metadata and picks from Sleeper (cached 20 s)."""
    client = SleeperClient(league_id=league_id)
    return client.get_draft(draft_id), client.get_draft_picks(draft_id)


@st.cache_data(ttl=300)
def _fetch_league_drafts(league_id: str) -> list[dict]:
    """Fetch all drafts for a league (cached 5 min)."""
    return SleeperClient(league_id=league_id).get_drafts()


def _render_live_draft_tab(
    dynasty_df: pd.DataFrame,
    sleeper_league_id: str,
    my_pick_slot: int,
    auto_refresh_enabled: bool,
    refresh_interval_secs: int,
) -> None:
    st.subheader("🏈 Live Draft Helper")

    if not sleeper_league_id:
        st.info(
            "👈 Enter your **Sleeper League ID** in the sidebar to activate the live draft helper.\n\n"
            "Find it in your Sleeper URL: `sleeper.app/leagues/<league_id>`"
        )
        return

    # Draft selector
    try:
        drafts = _fetch_league_drafts(sleeper_league_id)
    except Exception as exc:
        st.error(f"Could not load drafts for league `{sleeper_league_id}`: {exc}")
        return

    if not drafts:
        st.warning("No drafts found for this league.")
        return

    draft_labels = {
        f"{d.get('season', '?')} draft — {d.get('status', 'unknown')} (id: {d['draft_id']})": d["draft_id"]
        for d in drafts
    }
    selected_label = st.selectbox("Select draft", list(draft_labels.keys()))
    draft_id = draft_labels[selected_label]

    # Refresh controls
    col_status, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("🔄 Refresh"):
            _fetch_draft_state.clear()
            st.rerun()

    # Fetch live state
    try:
        draft_info, picks = _fetch_draft_state(sleeper_league_id, draft_id)
    except Exception as exc:
        st.error(f"Error fetching draft state: {exc}")
        return

    draft_settings = draft_info.get("settings", {})
    total_rounds = int(draft_settings.get("rounds", 15))
    total_teams = int(draft_settings.get("teams", 12))
    status = draft_info.get("status", "unknown")
    picks_made = len(picks)
    total_picks = total_rounds * total_teams
    current_pick_no = picks_made + 1
    current_round = (picks_made // total_teams) + 1
    pick_in_round = (picks_made % total_teams) + 1

    with col_status:
        st.caption(
            f"Status: **{status.upper()}** | "
            f"Round {min(current_round, total_rounds)}/{total_rounds} | "
            f"Pick {current_pick_no}/{total_picks}"
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Round", f"{min(current_round, total_rounds)} / {total_rounds}")
    m2.metric("Pick in round", f"{pick_in_round} / {total_teams}")
    m3.metric("Overall pick", f"{current_pick_no}")
    m4.metric("Picks made", f"{picks_made} / {total_picks}")

    st.divider()

    # Build set of drafted player names (normalised)
    drafted_names: set[str] = set()
    for pick in picks:
        meta = pick.get("metadata", {})
        full = f"{meta.get('first_name', '')} {meta.get('last_name', '')}".strip()
        if full:
            drafted_names.add(_normalize_name(full))

    # Filter dynasty rankings to available players
    available_df = dynasty_df[
        ~dynasty_df["player_name"].apply(_normalize_name).isin(drafted_names)
    ].copy()
    available_df = available_df.sort_values("dynasty_vorp", ascending=False).reset_index(drop=True)
    available_df.index += 1  # 1-based rank

    # My picks (identified by draft_slot matching pick slot)
    my_picks = [p for p in picks if str(p.get("draft_slot")) == str(my_pick_slot)]

    col_avail, col_roster = st.columns([2, 1])

    with col_avail:
        st.subheader("🎯 Top Available")
        pos_filter_live = st.multiselect(
            "Filter position",
            ["QB", "RB", "WR", "TE"],
            default=["QB", "RB", "WR", "TE"],
            key="live_pos_filter",
        )
        shown = available_df[available_df["position"].isin(pos_filter_live)].head(30)

        pos_colors = {"QB": "#dbeafe", "RB": "#dcfce7", "WR": "#fef9c3", "TE": "#fae8ff"}

        def _row_color(row):
            color = pos_colors.get(row["position"], "#ffffff")
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            shown[["player_name", "position", "age", "ppg", "dynasty_vorp"]]
            .rename(columns={"player_name": "Player", "position": "Pos",
                             "age": "Age", "ppg": "PPG", "dynasty_vorp": "Dynasty VORP"})
            .style.apply(_row_color, axis=1)
            .format({"PPG": "{:.1f}", "Dynasty VORP": "{:.1f}"}),
            use_container_width=True,
            height=520,
        )

    with col_roster:
        st.subheader(f"📋 My Roster (slot {my_pick_slot})")

        if my_picks:
            my_roster_df = pd.DataFrame([
                {
                    "Rd": p["round"],
                    "Pick": p["pick_no"],
                    "Player": (
                        f"{p['metadata'].get('first_name', '')} "
                        f"{p['metadata'].get('last_name', '')}".strip()
                    ),
                    "Pos": p["metadata"].get("position", ""),
                }
                for p in my_picks
            ])
            st.dataframe(my_roster_df, use_container_width=True, hide_index=True)

            # Positional need
            st.subheader("📊 Positional Need")
            pos_counts = my_roster_df["Pos"].value_counts().to_dict()
            starter_targets = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
            need_rows = []
            for pos, target in starter_targets.items():
                have = pos_counts.get(pos, 0)
                need_rows.append({
                    "Pos": pos,
                    "Have": have,
                    "Need": max(0, target - have),
                    "": "✅" if have >= target else "⚠️",
                })
            st.dataframe(pd.DataFrame(need_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No picks recorded for your slot yet.")

    # Auto-refresh (only while draft is live)
    if auto_refresh_enabled and status == "drafting":
        time.sleep(refresh_interval_secs)
        _fetch_draft_state.clear()
        st.rerun()


if __name__ == "__main__":
    main()
