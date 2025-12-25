import re
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="Shifting Narratives",
    page_icon="🌍",
    layout="centered"
)

# =====================================================
# CONSTANTS
# =====================================================
DATA_PATH = Path("data/events.csv")

REQUIRED_COLUMNS = ["event_name", "stakeholder", "text", "date"]
OPTIONAL_COLUMNS = ["entity", "source", "collection_method"]

TOP_N_KEYWORDS = 30
POS_THRESHOLD = 0.08
NEG_THRESHOLD = -0.08

# =====================================================
# SENTIMENT LEXICON (STARTER)
# =====================================================
POS_WORDS = {
    "progress", "success", "agreement", "cooperation", "growth",
    "innovation", "support", "stability", "unity", "hope",
    "benefit", "achievement", "positive", "safe", "responsible"
}

NEG_WORDS = {
    "crisis", "failure", "conflict", "risk", "fear",
    "harm", "controversy", "discrimination", "violence",
    "loss", "problem", "negative", "collapse", "threat"
}

# =====================================================
# TEXT PROCESSING
# =====================================================
STOPWORDS = {
    "the","and","a","to","of","in","on","for","with","as","at","by",
    "is","are","was","were","be","been","being","it","this","that",
    "from","or","but","not","we","they","you","i","he","she","them",
    "his","her","their","our","us","your","my","me"
}

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return [t for t in text.split() if len(t) > 2 and t not in STOPWORDS]

def extract_keywords(text: str, n=TOP_N_KEYWORDS) -> pd.DataFrame:
    tokens = tokenize(text)
    freq = Counter(tokens).most_common(n)
    return pd.DataFrame(freq, columns=["keyword", "freq"])

def polarity(word: str) -> str:
    if word in POS_WORDS:
        return "Positive"
    if word in NEG_WORDS:
        return "Negative"
    return "Neutral"

def sentiment_from_keywords(df_kw: pd.DataFrame) -> dict:
    df = df_kw.copy()
    df["polarity"] = df["keyword"].apply(polarity)

    pos = df[df["polarity"] == "Positive"]["freq"].sum()
    neg = df[df["polarity"] == "Negative"]["freq"].sum()
    neu = df[df["polarity"] == "Neutral"]["freq"].sum()

    total = pos + neg + neu
    score = 0 if total == 0 else (pos - neg) / total

    if score > POS_THRESHOLD:
        label = "Positive"
    elif score < NEG_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    coverage = 0 if total == 0 else (pos + neg) / total

    return {
        "score": score,
        "label": label,
        "pos": pos,
        "neg": neg,
        "neu": neu,
        "coverage": coverage,
    }

# =====================================================
# COLOR SCALE
# =====================================================
COLOR_SCALE = alt.Scale(
    domain=["Positive", "Neutral", "Negative"],
    range=["#2ca02c", "#7f7f7f", "#d62728"]
)

# =====================================================
# LOAD DATA
# =====================================================
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

if "dataset" not in st.session_state:
    st.session_state.dataset = load_data()

df = st.session_state.dataset.copy()

# =====================================================
# SIDEBAR — INPUT + SELECTION
# =====================================================
st.sidebar.title("Data Input")

input_mode = st.sidebar.radio(
    "Input method",
    ["Upload CSV", "Paste Article"]
)

if input_mode == "Upload CSV":
    file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if file is not None:
        df_new = pd.read_csv(file)
        missing = [c for c in REQUIRED_COLUMNS if c not in df_new.columns]
        if missing:
            st.sidebar.error(f"Missing columns: {missing}")
        else:
            st.session_state.dataset = df_new.copy()
            df = st.session_state.dataset.copy()
            st.sidebar.success("Dataset loaded")

else:
    st.sidebar.subheader("Add Article")
    event_name = st.sidebar.text_input("Event name")
    stakeholder = st.sidebar.selectbox("Stakeholder", ["Government", "Media", "Public"])
    entity = st.sidebar.text_input("Entity (optional)")
    source = st.sidebar.text_input("Source (optional)")
    date = st.sidebar.date_input("Date")
    text = st.sidebar.text_area("Paste article text")

    if st.sidebar.button("Add Article"):
        if text.strip():
            new_row = {
                "event_name": event_name,
                "stakeholder": stakeholder,
                "entity": entity,
                "source": source,
                "date": str(date),
                "text": text,
            }
            st.session_state.dataset = pd.concat(
                [st.session_state.dataset, pd.DataFrame([new_row])],
                ignore_index=True
            )
            df = st.session_state.dataset.copy()
            st.sidebar.success("Article added")

# =====================================================
# SIDEBAR — SELECTION CONTROLS (MOVED HERE)
# =====================================================
st.sidebar.title("View Controls")

if df.empty:
    st.warning("No data available.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["text"] = df["text"].astype(str)

selected_event = st.sidebar.selectbox(
    "Select Event",
    sorted(df["event_name"].unique())
)

selected_stakeholder = st.sidebar.selectbox(
    "Select Stakeholder",
    sorted(df["stakeholder"].unique())
)

event_df = df[df["event_name"] == selected_event]
stake_df = event_df[event_df["stakeholder"] == selected_stakeholder]

entities = ["All"] + sorted(
    e for e in stake_df["entity"].dropna().unique() if str(e).strip()
)

selected_entity = st.sidebar.selectbox("Select Entity", entities)

if selected_entity != "All":
    stake_df = stake_df[stake_df["entity"] == selected_entity]

# =====================================================
# MAIN DASHBOARD
# =====================================================
st.title("Shifting Narratives")
st.caption("Explainable sentiment analysis using keyword frequency")

row = stake_df.sort_values("date").iloc[-1]

kw_df = extract_keywords(row["text"])
stats = sentiment_from_keywords(kw_df)
kw_df["polarity"] = kw_df["keyword"].apply(polarity)

color = (
    "#2ca02c" if stats["label"] == "Positive"
    else "#d62728" if stats["label"] == "Negative"
    else "#7f7f7f"
)

st.subheader(f"{selected_stakeholder} — {selected_event}")

st.markdown(
    f"""
<div style='padding:18px;border-radius:10px;background:#f5f5f5;font-size:18px;'>
<b>Article excerpt:</b><br>
{row['text'][:800]}{"..." if len(row['text']) > 800 else ""}
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    f"<div style='font-size:18px;color:{color};margin-top:10px;'>"
    f"Sentiment score: {stats['score']:.2f} ({stats['label']})</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# WHY THIS SCORE
# =====================================================
st.markdown("## Why this sentiment?")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Positive freq", stats["pos"])
m2.metric("Negative freq", stats["neg"])
m3.metric("Neutral freq", stats["neu"])
m4.metric("Coverage", f"{stats['coverage']*100:.0f}%")

top_kw = kw_df.sort_values("freq", ascending=False).head(15)

bar = (
    alt.Chart(top_kw)
    .mark_bar()
    .encode(
        x=alt.X("freq:Q", title="Frequency"),
        y=alt.Y("keyword:N", sort="-x"),
        color=alt.Color("polarity:N", scale=COLOR_SCALE),
        tooltip=["keyword", "freq", "polarity"]
    )
    .properties(height=420, title="Top Keywords by Frequency and Sentiment")
)

st.altair_chart(bar, use_container_width=True)

st.markdown(
    """
🟢 Positive  ⚪ Neutral  🔴 Negative  

Bar length shows how often a keyword appears.
Color shows whether it contributes positively, negatively, or neutrally.
"""
)

# =====================================================
# OVERVIEW — STAKEHOLDERS
# =====================================================
st.markdown("---")
st.markdown("## Stakeholder Overview (Selected Event)")

def avg_sentiment(texts):
    scores = []
    for t in texts:
        kw = extract_keywords(t)
        scores.append(sentiment_from_keywords(kw)["score"])
    return sum(scores) / len(scores) if scores else 0

overview = (
    event_df.groupby("stakeholder")["text"]
    .apply(avg_sentiment)
    .reset_index(name="avg_score")
)

overview["category"] = overview["avg_score"].apply(
    lambda s: "Positive" if s > POS_THRESHOLD else "Negative" if s < NEG_THRESHOLD else "Neutral"
)

overview_chart = (
    alt.Chart(overview)
    .mark_bar()
    .encode(
        x="stakeholder:N",
        y=alt.Y("avg_score:Q", scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color("category:N", scale=COLOR_SCALE),
        tooltip=["stakeholder", alt.Tooltip("avg_score:Q", format=".2f")]
    )
    .properties(height=260)
)

st.altair_chart(overview_chart, use_container_width=True)

st.markdown("---")
if st.button("Done Exploring 🎉"):
    st.balloons()
