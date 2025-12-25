import re
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Shifting Narratives", page_icon="🌍", layout="centered")


# =========================
# CONFIGURATION
# =========================
DATA_PATH = Path("data/events.csv")

REQUIRED_COLUMNS = ["event_name", "stakeholder", "text", "date"]
OPTIONAL_COLUMNS = ["entity", "source"]

# Keyword extraction settings
TOP_N_KEYWORDS = 40

# Sentiment thresholds (after normalization)
POS_THRESHOLD = 0.08
NEG_THRESHOLD = -0.08


# =========================
# SIMPLE LEXICON (EDITABLE)
# =========================
# NOTE: This is a starter dictionary for the prototype.
# You should expand it over time or allow users to upload their own lexicon CSV.
POS_WORDS = {
    "progress", "success", "agreement", "cooperation", "improve", "growth", "peace",
    "innovation", "safe", "stability", "support", "unity", "celebration", "solution",
    "achievement", "positive", "hope", "benefit", "win", "responsible"
}

NEG_WORDS = {
    "crisis", "failure", "scandal", "conflict", "risk", "fear", "harm", "danger",
    "corruption", "hate", "discrimination", "violence", "controversy", "loss",
    "problem", "negative", "attack", "collapse", "chaos", "threat"
}


# =========================
# TEXT PROCESSING UTILITIES
# =========================
STOPWORDS = {
    "the", "and", "a", "an", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these",
    "those", "from", "or", "but", "not", "we", "they", "you", "i", "he", "she", "them",
    "his", "her", "their", "our", "us", "your", "my", "me", "will", "would", "can",
    "could", "should", "may", "might", "about", "into", "over", "after", "before",
    "more", "most", "some", "any", "no", "yes", "if", "then", "than", "so", "such",
    "also", "very", "just", "up", "down", "out", "now", "new", "one", "two", "many",
    "all"
}

def clean_and_tokenize(text: str) -> list[str]:
    """TURN RAW TEXT INTO A LIST OF MEANINGFUL WORD TOKENS."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)          # REMOVE URLS
    text = re.sub(r"[^a-z\s]", " ", text)         # REMOVE PUNCTUATION / NUMBERS
    tokens = [t for t in text.split() if len(t) >= 3 and t not in STOPWORDS]
    return tokens

def extract_keywords_and_freq(text: str, top_n: int = TOP_N_KEYWORDS) -> pd.DataFrame:
    """EXTRACT TOP-N WORDS WITH THEIR FREQUENCIES."""
    tokens = clean_and_tokenize(text)
    counts = Counter(tokens)
    common = counts.most_common(top_n)
    return pd.DataFrame(common, columns=["keyword", "freq"])

def keyword_polarity(word: str) -> str:
    """CLASSIFY WORDS BY LEXICON POLARITY."""
    if word in POS_WORDS:
        return "Positive"
    if word in NEG_WORDS:
        return "Negative"
    return "Neutral"

def compute_sentiment_from_keywords(kw_df: pd.DataFrame) -> dict:
    """
    COMPUTE A SENTIMENT SCORE USING FREQUENCY-WEIGHTED KEYWORDS.
    SCORE RANGE: APPROXIMATELY -1 TO +1.
    """
    if kw_df.empty:
        return {
            "sentiment_score": 0.0,
            "sentiment_category": "Neutral",
            "pos_sum": 0,
            "neg_sum": 0,
            "neu_sum": 0,
            "coverage": 0.0,
        }

    tmp = kw_df.copy()
    tmp["polarity"] = tmp["keyword"].apply(keyword_polarity)

    pos_sum = int(tmp.loc[tmp["polarity"] == "Positive", "freq"].sum())
    neg_sum = int(tmp.loc[tmp["polarity"] == "Negative", "freq"].sum())
    neu_sum = int(tmp.loc[tmp["polarity"] == "Neutral", "freq"].sum())

    total = pos_sum + neg_sum + neu_sum
    score = 0.0 if total == 0 else (pos_sum - neg_sum) / total

    # COVERAGE = HOW MUCH OF THE TOP KEYWORDS ARE IN POS/NEG (NOT NEUTRAL)
    emotional_total = pos_sum + neg_sum
    coverage = 0.0 if total == 0 else emotional_total / total

    if score > POS_THRESHOLD:
        cat = "Positive"
    elif score < NEG_THRESHOLD:
        cat = "Negative"
    else:
        cat = "Neutral"

    return {
        "sentiment_score": float(score),
        "sentiment_category": cat,
        "pos_sum": pos_sum,
        "neg_sum": neg_sum,
        "neu_sum": neu_sum,
        "coverage": float(coverage),
    }


# =========================
# VISUAL SETTINGS
# =========================
SENTIMENT_COLOR_SCALE = alt.Scale(
    domain=["Positive", "Neutral", "Negative"],
    range=["#2ca02c", "#7f7f7f", "#d62728"]
)


# =========================
# DATA TEMPLATE DOWNLOAD
# =========================
def template_csv_bytes() -> bytes:
    template = pd.DataFrame(
        {
            "event_name": ["Example Event"],
            "stakeholder": ["Government"],
            "entity": ["Ministry of Environment"],
            "source": ["https://example.com/article"],
            "date": ["2025-01-01"],
            "text": ["Paste your article text here..."],
        }
    )
    return template.to_csv(index=False).encode("utf-8")


# =========================
# LOAD BASE DATA (IF EXISTS)
# =========================
def load_base_df() -> pd.DataFrame:
    if DATA_PATH.exists():
        df0 = pd.read_csv(DATA_PATH)
        return df0
    # FALLBACK DATA
    return pd.DataFrame(
        {
            "event_name": ["Climate Summit", "Climate Summit", "Climate Summit"],
            "stakeholder": ["Government", "Media", "Public"],
            "entity": ["Gov Office", "News Outlet", "Online Users"],
            "source": ["", "", ""],
            "date": ["2023-11-15", "2023-11-15", "2023-11-15"],
            "text": [
                "Government stresses cooperation and long-term climate agreements and progress.",
                "Media emphasizes mixed reactions, controversy, and risk in negotiation outcomes.",
                "Public expresses hope but also frustration about slow progress and crisis concerns.",
            ],
        }
    )


# =========================
# SESSION STATE DATASET
# =========================
if "dataset" not in st.session_state:
    st.session_state.dataset = load_base_df()

df = st.session_state.dataset.copy()


# =========================
# SIDEBAR: INPUT METHODS
# =========================
st.sidebar.title("Data Input")

input_mode = st.sidebar.radio(
    "Choose input method",
    ["Upload CSV (Template)", "Paste Article (Add Button)"],
    index=0
)

st.sidebar.download_button(
    label="Download CSV Template",
    data=template_csv_bytes(),
    file_name="events_template.csv",
    mime="text/csv"
)

if input_mode == "Upload CSV (Template)":
    uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            missing = [c for c in REQUIRED_COLUMNS if c not in uploaded_df.columns]
            if missing:
                st.sidebar.error(f"Missing required columns: {missing}")
            else:
                st.session_state.dataset = uploaded_df.copy()
                df = st.session_state.dataset.copy()
                st.sidebar.success("CSV loaded successfully.")
        except Exception as e:
            st.sidebar.error("Could not read CSV. Please check formatting.")

else:
    st.sidebar.subheader("Add Article")
    event_name = st.sidebar.text_input("Event name", value="New Event")
    stakeholder = st.sidebar.selectbox("Stakeholder", ["Government", "Media", "Public"])
    entity = st.sidebar.text_input("Entity (optional)", value="")
    source = st.sidebar.text_input("Source (optional)", value="")
    date = st.sidebar.date_input("Date")
    text = st.sidebar.text_area("Paste article text here", height=200)

    if st.sidebar.button("Add Article"):
        if not text.strip():
            st.sidebar.error("Please paste an article text.")
        else:
            new_row = {
                "event_name": event_name.strip(),
                "stakeholder": stakeholder,
                "entity": entity.strip(),
                "source": source.strip(),
                "date": str(date),
                "text": text.strip(),
            }
            st.session_state.dataset = pd.concat(
                [st.session_state.dataset, pd.DataFrame([new_row])],
                ignore_index=True
            )
            df = st.session_state.dataset.copy()
            st.sidebar.success("Article added.")


# =========================
# VALIDATE / CLEAN DATA
# =========================
# KEEP ONLY COLUMNS WE KNOW + REQUIRED
for col in REQUIRED_COLUMNS:
    if col not in df.columns:
        df[col] = ""

for col in OPTIONAL_COLUMNS:
    if col not in df.columns:
        df[col] = ""

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()

# ENSURE TEXT IS STRING
df["text"] = df["text"].astype(str)


# =========================
# COMPUTE KEYWORDS + SENTIMENT FOR EACH ROW
# =========================
# NOTE: FOR LARGE DATASETS THIS SHOULD BE CACHED OR BATCHED.
keyword_strings = []
sent_scores = []
sent_cats = []
pos_sums = []
neg_sums = []
neu_sums = []
coverages = []

for t in df["text"].tolist():
    kw_df = extract_keywords_and_freq(t, TOP_N_KEYWORDS)
    stats = compute_sentiment_from_keywords(kw_df)

    keyword_strings.append(", ".join(kw_df["keyword"].tolist()[:15]))
    sent_scores.append(stats["sentiment_score"])
    sent_cats.append(stats["sentiment_category"])
    pos_sums.append(stats["pos_sum"])
    neg_sums.append(stats["neg_sum"])
    neu_sums.append(stats["neu_sum"])
    coverages.append(stats["coverage"])

df["keywords"] = keyword_strings
df["sentiment_score"] = sent_scores
df["sentiment_category"] = sent_cats
df["pos_sum"] = pos_sums
df["neg_sum"] = neg_sums
df["neu_sum"] = neu_sums
df["coverage"] = coverages


# =========================
# MAIN UI HEADER
# =========================
st.title("Shifting Narratives")
st.caption("Compare narratives across perspectives using keyword frequency and explainable sentiment scoring.")

st.markdown("### What This App Needs (CSV Columns)")
st.write(
    "**Required:** event_name, stakeholder, text, date  |  "
    "**Optional:** entity, source  |  "
    "The app extracts keywords automatically and computes sentiment from them."
)

st.markdown("---")


# =========================
# CONTROLS: EVENT + STAKEHOLDER + OPTIONAL ENTITY
# =========================
left, right = st.columns([1, 1])

with left:
    selected_event = st.selectbox("Select Event", sorted(df["event_name"].unique()))

with right:
    selected_stakeholder = st.selectbox("Select Stakeholder", sorted(df["stakeholder"].unique()))

event_df = df[df["event_name"] == selected_event].copy()
stake_df = event_df[event_df["stakeholder"] == selected_stakeholder].copy()

entities = ["(All)"] + sorted([e for e in stake_df["entity"].unique() if str(e).strip() != "" and str(e) != "nan"])
selected_entity = st.selectbox("Select Entity (optional)", entities)

if selected_entity != "(All)":
    stake_df = stake_df[stake_df["entity"] == selected_entity].copy()

# SELECT A DOCUMENT
stake_df = stake_df.sort_values("date")
doc_options = stake_df.index.tolist()
doc_label_map = {
    idx: f"{stake_df.loc[idx,'date'].date()} — {str(stake_df.loc[idx,'entity'])[:30]}"
    for idx in doc_options
}
selected_doc_idx = st.selectbox(
    "Select Document Example",
    options=doc_options,
    format_func=lambda x: doc_label_map.get(x, str(x))
)

row = stake_df.loc[selected_doc_idx]


# =========================
# SUMMARY PANEL
# =========================
score = float(row["sentiment_score"])
cat = row["sentiment_category"]

color = "#2ca02c" if cat == "Positive" else "#d62728" if cat == "Negative" else "#7f7f7f"

st.subheader(f"{selected_stakeholder} Perspective — {selected_event}")

st.markdown(
    f"""
<div style='padding:18px;border-radius:12px;background:#f5f5f5;font-size:20px;line-height:1.5;'>
  <b>Document Text (Example):</b><br>
  {row['text'][:900]}{"..." if len(row['text']) > 900 else ""}
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    f"<div style='margin-top:10px;color:{color};font-weight:bold;font-size:18px;'>"
    f"Sentiment: {score:.2f} ({cat})</div>",
    unsafe_allow_html=True
)

st.markdown(
    f"<div style='font-size:16px;margin-top:8px;'><b>Top keywords:</b> {row['keywords']}</div>",
    unsafe_allow_html=True
)

st.markdown("### Color Legend (Placed Near The Results)")
st.write("🟢 Positive   |   ⚪ Neutral   |   🔴 Negative")


# =========================
# EXPLANATION: WHY THIS SCORE?
# =========================
st.markdown("---")
st.markdown("## Why This Sentiment Score?")

# REBUILD KEYWORDS FOR THE SELECTED DOCUMENT FOR EXPLANATION
kw_df = extract_keywords_and_freq(row["text"], TOP_N_KEYWORDS)
kw_df["polarity"] = kw_df["keyword"].apply(keyword_polarity)

stats = compute_sentiment_from_keywords(kw_df)

explain_cols = st.columns(4)
explain_cols[0].metric("Positive freq", stats["pos_sum"])
explain_cols[1].metric("Negative freq", stats["neg_sum"])
explain_cols[2].metric("Neutral freq", stats["neu_sum"])
explain_cols[3].metric("Coverage", f"{stats['coverage']*100:.0f}%")

st.caption("Coverage shows how much of the top keywords are emotional (positive/negative) instead of neutral.")

# TOP CONTRIBUTORS
top_pos = kw_df[kw_df["polarity"] == "Positive"].sort_values("freq", ascending=False).head(10)
top_neg = kw_df[kw_df["polarity"] == "Negative"].sort_values("freq", ascending=False).head(10)

contrib_df = pd.concat(
    [
        top_pos.assign(type="Top Positive"),
        top_neg.assign(type="Top Negative")
    ],
    ignore_index=True
)

if contrib_df.empty:
    st.info("No sentiment keywords were found in this document. The score stays Neutral.")
else:
    bar = (
        alt.Chart(contrib_df)
        .mark_bar()
        .encode(
            x=alt.X("freq:Q", title="Frequency"),
            y=alt.Y("keyword:N", sort="-x", title="Keyword"),
            color=alt.Color("polarity:N", scale=SENTIMENT_COLOR_SCALE),
            tooltip=["keyword", "freq", "polarity"]
        )
        .properties(height=380, title="Top Keyword Contributions")
    )
    st.altair_chart(bar, use_container_width=True)

st.markdown(
    f"""
**Score formula (frequency-weighted):**  
`score = (positive_freq - negative_freq) / (positive_freq + negative_freq + neutral_freq)`  

This document is **{cat}** because positive keywords appear {stats["pos_sum"]} times
and negative keywords appear {stats["neg_sum"]} times in the top keywords.
"""
)


# =========================
# KEYWORD CLOUD (ALTair TEXT-BASED)
# =========================
st.markdown("---")
st.markdown("## Keyword Cloud (Frequency-Based)")

cloud_df = kw_df.copy()
cloud_df["size"] = cloud_df["freq"] * 4 + 12  # SIMPLE SCALING
cloud_df["x"] = range(len(cloud_df))
cloud_df["y"] = (cloud_df["x"] % 6)

cloud_chart = (
    alt.Chart(cloud_df)
    .mark_text()
    .encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q", axis=None),
        text="keyword:N",
        size=alt.Size("size:Q", legend=None),
        color=alt.Color("polarity:N", scale=SENTIMENT_COLOR_SCALE),
        tooltip=["keyword", "freq", "polarity"]
    )
    .properties(height=220)
)
st.altair_chart(cloud_chart, use_container_width=True)
st.caption("Word size shows frequency. Word color shows polarity (positive/neutral/negative).")


# =========================
# OVERVIEW: INSTITUTION / STAKEHOLDER SENTIMENT
# =========================
st.markdown("---")
st.markdown("## Overview: Average Sentiment By Stakeholder (Selected Event)")

overview = (
    event_df.groupby("stakeholder", as_index=False)["sentiment_score"]
    .mean()
    .rename(columns={"sentiment_score": "avg_sentiment"})
)
overview["sentiment_category"] = overview["avg_sentiment"].apply(
    lambda s: "Positive" if s > POS_THRESHOLD else "Negative" if s < NEG_THRESHOLD else "Neutral"
)

overview_chart = (
    alt.Chart(overview)
    .mark_bar()
    .encode(
        x=alt.X("stakeholder:N", title="Stakeholder"),
        y=alt.Y("avg_sentiment:Q", title="Average Sentiment", scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color("sentiment_category:N", scale=SENTIMENT_COLOR_SCALE),
        tooltip=["stakeholder", alt.Tooltip("avg_sentiment:Q", format=".2f")]
    )
    .properties(height=280)
)
st.altair_chart(overview_chart, use_container_width=True)


# =========================
# OPTIONAL: ENTITY BREAKDOWN
# =========================
st.markdown("### Optional: Entity Breakdown (If Entities Exist)")
if event_df["entity"].astype(str).str.strip().replace("nan", "").eq("").all():
    st.info("No entity values found in this dataset. Add an 'entity' column to enable this view.")
else:
    entity_overview = (
        event_df[event_df["entity"].astype(str).str.strip().replace("nan", "") != ""]
        .groupby(["stakeholder", "entity"], as_index=False)["sentiment_score"]
        .mean()
        .rename(columns={"sentiment_score": "avg_sentiment"})
    )
    entity_overview["sentiment_category"] = entity_overview["avg_sentiment"].apply(
        lambda s: "Positive" if s > POS_THRESHOLD else "Negative" if s < NEG_THRESHOLD else "Neutral"
    )

    entity_chart = (
        alt.Chart(entity_overview)
        .mark_bar()
        .encode(
            x=alt.X("entity:N", sort="-y", title="Entity"),
            y=alt.Y("avg_sentiment:Q", title="Average Sentiment", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("sentiment_category:N", scale=SENTIMENT_COLOR_SCALE),
            column=alt.Column("stakeholder:N", title="Stakeholder"),
            tooltip=["stakeholder", "entity", alt.Tooltip("avg_sentiment:Q", format=".2f")]
        )
        .properties(height=240)
    )
    st.altair_chart(entity_chart, use_container_width=True)


# =========================
# TIMELINE: SELECTED EVENT ONLY
# =========================
st.markdown("---")
st.markdown("## Timeline: Sentiment Over Time (Selected Event)")

timeline_chart = (
    alt.Chart(event_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y", labelAngle=-35)),
        y=alt.Y("sentiment_score:Q", title="Sentiment Score", scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color("stakeholder:N", title="Stakeholder"),
        tooltip=["event_name", "stakeholder", "entity", "sentiment_score", "date"]
    )
    .properties(height=260)
)
st.altair_chart(timeline_chart, use_container_width=True)

st.markdown("---")
if st.button("Done Exploring 🎉"):
    st.balloons()
