import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.nfl_stats import NFLStats
from src.models.vorp import LeagueSettings, VORPCalculator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Dynasty Rankings",
            "📈 Aging Curves",
            "⚖️ Positional Scarcity",
            "🔍 Player Comparison",
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


if __name__ == "__main__":
    main()
