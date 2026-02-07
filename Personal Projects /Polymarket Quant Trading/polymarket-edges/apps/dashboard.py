"""Streamlit dashboard for Polymarket Edges."""

import json

import streamlit as st

from polymarket_edges.db import Database


# Page config
st.set_page_config(
    page_title="Polymarket Edges",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_database():
    """Get database connection (cached)."""
    return Database()


def format_score_color(score: float) -> str:
    """Return color based on score value."""
    if score >= 70:
        return "🟢"
    elif score >= 40:
        return "🟡"
    else:
        return "🔴"


def main():
    """Main dashboard application."""

    # Header
    st.title("📊 Polymarket Edges")
    st.markdown("**Analytics for Polymarket prediction markets** - Informational only, no trading")

    # Warning banner
    st.markdown(
        """
        <div class="warning-box">
        ⚠️ <strong>Disclaimer:</strong> This tool is for informational and analytical purposes only.
        It does not provide financial advice and does not execute any trades. Always review Polymarket's
        terms of service and comply with applicable laws in your jurisdiction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar filters
    st.sidebar.header("Filters")

    db = get_database()

    # Load data
    try:
        scores_df = db.get_latest_scores(limit=1000)

        if scores_df.empty:
            st.warning(
                "No scored data found. Please run the pipeline first:\n\n"
                "```bash\n"
                "polymarket-edges pipeline\n"
                "```"
            )
            return

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Sidebar filters
    min_combined = st.sidebar.slider(
        "Minimum Combined Score",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
    )

    max_spread = st.sidebar.slider(
        "Maximum Spread",
        min_value=0.0,
        max_value=0.5,
        value=0.5,
        step=0.01,
        format="%.2f",
    )

    max_risk = st.sidebar.slider(
        "Maximum Rules Risk Score",
        min_value=0,
        max_value=100,
        value=100,
        step=5,
    )

    search_query = st.sidebar.text_input("Search markets", "")

    # Apply filters
    filtered_df = scores_df[
        (scores_df["combined_score"] >= min_combined)
        & (scores_df["rules_risk_score"] <= max_risk)
    ]

    if max_spread < 0.5:
        filtered_df = filtered_df[
            (filtered_df["spread"].isna()) | (filtered_df["spread"] <= max_spread)
        ]

    if search_query:
        filtered_df = filtered_df[
            filtered_df["question"].str.contains(search_query, case=False, na=False)
        ]

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Markets", len(scores_df))

    with col2:
        st.metric("Filtered Markets", len(filtered_df))

    with col3:
        if not filtered_df.empty:
            avg_combined = filtered_df["combined_score"].mean()
            st.metric("Avg Combined Score", f"{avg_combined:.1f}")
        else:
            st.metric("Avg Combined Score", "N/A")

    with col4:
        if not filtered_df.empty:
            avg_risk = filtered_df["rules_risk_score"].mean()
            st.metric("Avg Rules Risk", f"{avg_risk:.1f}")
        else:
            st.metric("Avg Rules Risk", "N/A")

    # Main table
    st.header("Ranked Markets")

    if filtered_df.empty:
        st.info("No markets match the current filters.")
        return

    # Prepare display dataframe
    display_df = filtered_df.copy()
    display_df["score_indicator"] = display_df["combined_score"].apply(format_score_color)

    # Select columns to display
    display_columns = [
        "score_indicator",
        "question",
        "outcome",
        "mid_price",
        "spread",
        "tradability_score",
        "rules_risk_score",
        "combined_score",
    ]

    # Rename for display
    column_config = {
        "score_indicator": st.column_config.TextColumn("", width="small"),
        "question": st.column_config.TextColumn("Market Question", width="large"),
        "outcome": st.column_config.TextColumn("Outcome", width="small"),
        "mid_price": st.column_config.NumberColumn("Mid Price", format="%.3f"),
        "spread": st.column_config.NumberColumn("Spread", format="%.4f"),
        "tradability_score": st.column_config.NumberColumn("Trade Score", format="%.1f"),
        "rules_risk_score": st.column_config.NumberColumn("Risk Score", format="%.1f"),
        "combined_score": st.column_config.NumberColumn("Combined", format="%.1f"),
    }

    # Display table with selection
    st.dataframe(
        display_df[display_columns],
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    # Detailed view section
    st.header("Market Details")

    # Let user select a market to view details
    selected_question = st.selectbox(
        "Select a market to view details:",
        options=filtered_df["question"].unique(),
        index=0,
    )

    if selected_question:
        market_data = filtered_df[filtered_df["question"] == selected_question].iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Scoring Breakdown")
            st.metric("Combined Score", f"{market_data['combined_score']:.1f}/100")
            st.metric("Tradability Score", f"{market_data['tradability_score']:.1f}/100")
            st.metric("Rules Risk Score", f"{market_data['rules_risk_score']:.1f}/100")

            if market_data["mid_price"] is not None:
                st.metric("Mid Price", f"{market_data['mid_price']:.3f}")
            if market_data["spread"] is not None:
                st.metric("Spread", f"{market_data['spread']:.4f}")

        with col2:
            st.subheader("Rules Analysis")

            if market_data.get("resolution_source"):
                st.markdown(f"**Resolution Source:** {market_data['resolution_source']}")

            if market_data.get("ambiguity_reasons"):
                try:
                    reasons = json.loads(market_data["ambiguity_reasons"])
                    if reasons:
                        st.markdown("**Ambiguity Concerns:**")
                        for reason in reasons:
                            st.markdown(f"- {reason}")
                except (json.JSONDecodeError, TypeError):
                    pass

        # Full question display
        st.subheader("Market Question")
        st.markdown(f"_{selected_question}_")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>Polymarket Edges - Analytics Only | No Trading Automation</p>
        <p>Data freshness depends on when you last ran update-quotes and score commands</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
