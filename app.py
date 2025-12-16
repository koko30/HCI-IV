import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(
    page_title="Shifting Narratives",
    page_icon="🌍",
    layout="centered"
)

# -------------------------------------------------
# Data Loading
# -------------------------------------------------
data_path = Path("data/events.csv")

if data_path.exists():
    df = pd.read_csv(data_path)
else:
    df = pd.DataFrame({
        "event_name": [
            "Climate Summit", "Climate Summit", "Climate Summit",
            "AI Regulation", "AI Regulation", "AI Regulation",
            "World Cup Final", "World Cup Final", "World Cup Final"
        ],
        "stakeholder": [
            "Government", "Media", "Public",
            "Government", "Media", "Public",
            "Government", "Media", "Public"
        ],
        "text_summary": [
            "Government stresses cooperation and long-term climate agreements.",
            "Media emphasizes mixed reactions and different negotiation outcomes.",
            "Public expresses hope but also frustration about slow progress.",
            "Government highlights AI innovation and safety frameworks.",
            "Media reports on heated debates about regulation speed.",
            "Public discussions show uncertainty and cautious optimism.",
            "Government congratulates athletes and focuses on national pride.",
            "Media covers intense match moments and controversial decisions.",
            "Public reactions include excitement and a calm appreciation of the match."
        ],
        "sentiment": [
            0.45, 0.02, -0.18,
            0.32, -0.22, 0.00,
            0.52, -0.14, 0.03
        ],
        "keywords": [
            "agreement, targets, cooperation",
            "reactions, negotiation, outcomes",
            "hope, frustration, progress",
            "innovation, safety, framework",
            "risk, debate, speed",
            "uncertainty, optimism, discussion",
            "pride, celebration, unity",
            "drama, controversy, intensity",
            "reaction, appreciation, calm"
        ],
        "date": [
            "2023-11-15", "2023-11-15", "2023-11-15",
            "2024-02-10", "2024-02-10", "2024-02-10",
            "2022-12-18", "2022-12-18", "2022-12-18"
        ]
    })

df["date"] = pd.to_datetime(df["date"])

# -------------------------------------------------
# Sentiment Categorization
# -------------------------------------------------
def label_sentiment(s):
    if s > 0.05:
        return "Positive"
    elif s < -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment_category"] = df["sentiment"].apply(label_sentiment)

sentiment_color_scale = alt.Scale(
    domain=["Positive", "Neutral", "Negative"],
    range=["#2ca02c", "#7f7f7f", "#d62728"]
)

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("Controls")

event = st.sidebar.selectbox(
    "Select Event",
    options=df["event_name"].unique()
)

stakeholder_choice = st.sidebar.radio(
    "Select Stakeholder",
    options=df["stakeholder"].unique()
)

mood_adjust = st.sidebar.slider(
    "Adjust Sentiment Intensity",
    -0.5, 0.5, 0.0, 0.05
)

comparison_mode = st.sidebar.checkbox("Enable Comparison Mode")

# -------------------------------------------------
# Filter Event Data
# -------------------------------------------------
event_df = df[df["event_name"] == event].copy()
event_df["sentiment"] = event_df["sentiment"] + mood_adjust
event_df["sentiment_category"] = event_df["sentiment"].apply(label_sentiment)

if stakeholder_choice in event_df["stakeholder"].unique():
    row = event_df[event_df["stakeholder"] == stakeholder_choice].iloc[0]
else:
    row = event_df.iloc[0]

row_sentiment = float(row["sentiment"])
row_category = label_sentiment(row_sentiment)

# -------------------------------------------------
# Title + Summary
# -------------------------------------------------
st.title("Shifting Narratives")
st.subheader(f"{stakeholder_choice} Perspective On “{event}”")

sent_color = (
    "#2ca02c" if row_category == "Positive"
    else "#d62728" if row_category == "Negative"
    else "#7f7f7f"
)

st.markdown(
    f"""
<div style='padding:18px;border-radius:10px;background:#f5f5f5;font-size:22px;'>
  <b>Summary ({stakeholder_choice}):</b><br>{row['text_summary']}
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='margin-top:10px;color:{sent_color};font-weight:bold;font-size:20px;'>Sentiment: {row_sentiment:.2f} ({row_category})</div>",
    unsafe_allow_html=True
)

st.markdown(f"<div style='font-size:18px;margin-top:10px;'><b>Keywords:</b> {row['keywords']}</div>", unsafe_allow_html=True)

st.markdown(
    """
### Color Meaning  
🟢 Green = Positive  
⚪ Grey = Neutral  
🔴 Red = Negative
"""
)

# -------------------------------------------------
# Time Chart (All Events)
# -------------------------------------------------
st.markdown("### Sentiment Over Time (All Events)")

time_chart = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y", labelAngle=-45)),
        y=alt.Y("sentiment:Q", title="Sentiment Score", scale=alt.Scale(domain=[-0.6, 0.6])),
        color=alt.Color("stakeholder:N"),
        tooltip=["event_name", "stakeholder", "sentiment", "date"]
    )
)

st.altair_chart(time_chart, use_container_width=True)

# -------------------------------------------------
# Comparison Mode
# -------------------------------------------------
if comparison_mode:

    st.markdown("### Sentiment Comparison For Selected Event")

    # Basic bar chart
    bar_chart = (
        alt.Chart(event_df)
        .mark_bar()
        .encode(
            x="stakeholder:N",
            y=alt.Y("sentiment:Q", scale=alt.Scale(domain=[-0.6, 0.6])),
            color=alt.Color("sentiment_category:N", scale=sentiment_color_scale),
            tooltip=["stakeholder", "sentiment", "sentiment_category"]
        )
    )

    st.altair_chart(bar_chart, use_container_width=True)

    # -------------------------------------------------
    # NEW CHART: Convert Sentiment Weight Into Percentages
    # -------------------------------------------------
    st.markdown("### Overall Emotional Weight")

    weight_df = event_df.copy()
    weight_df["intensity"] = weight_df["sentiment"].abs()

    weight_sums = weight_df.groupby("sentiment_category")["intensity"].sum().reset_index()
    total = weight_sums["intensity"].sum()

    weight_sums["percent"] = weight_sums["intensity"] / total

    percent_chart = (
        alt.Chart(weight_sums)
        .mark_bar()
        .encode(
            x=alt.X("sentiment_category:N", title="Sentiment"),
            y=alt.Y("percent:Q", title="Percentage", axis=alt.Axis(format="%")),
            color=alt.Color("sentiment_category:N", scale=sentiment_color_scale),
            tooltip=[
                "sentiment_category:N",
                alt.Tooltip("percent:Q", format=".1%", title="Percent"),
                "intensity:Q"
            ]
        )
        .properties(width=500, height=260)
    )

    st.altair_chart(percent_chart, use_container_width=True)

    st.markdown(
        """
        <div style='font-size:16px;'>
        This chart shows how much of the total emotional intensity is Positive,
        Neutral, or Negative.  
        Values are converted into percentages, so they always sum to 100%.
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# End Interaction
# -------------------------------------------------
st.markdown("---")
if st.button("Done Exploring 🎉"):
    st.balloons()
