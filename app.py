import re
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from textblob import TextBlob


# =====================================================
# Page Setup
# =====================================================
st.set_page_config(
    page_title="Shifting Narratives",
    page_icon="🌍",
    layout="centered"
)

# =====================================================
# Configuration
# =====================================================
DATA_PATH = Path("data/events.csv")

REQUIRED_COLUMNS = ["event_name", "stakeholder", "text", "date"]
OPTIONAL_COLUMNS = ["entity", "source", "collection_method"]

TOP_N_KEYWORDS = 30
POS_THRESHOLD = 0.08
NEG_THRESHOLD = -0.08


# =====================================================
# Sentiment Lexicons (Explainable)
# =====================================================
POS_WORDS = {
    "progress","success","agreement","cooperation","growth","innovation",
    "support","stability","unity","hope","benefit","achievement","positive",
    "safe","responsible","improve","improved","solution","trust","peace"
}

NEG_WORDS = {
    "crisis","failure","conflict","risk","fear","harm","controversy",
    "discrimination","violence","loss","problem","negative","collapse",
    "threat","chaos","scandal","attack","hate"
}

STOPWORDS = {
    "the","and","a","to","of","in","on","for","with","as","at","by",
    "is","are","was","were","be","been","being","it","this","that",
    "from","or","but","not","we","they","you","i","he","she","them",
    "his","her","their","our","us","your","my","me","will","would",
    "can","could","should","may","might","about","into","over","after",
    "before","more","most","some","any","no","yes","if","then","than",
    "so","such","also","very","just","up","down","out","now","new"
}


# =====================================================
# Text Processing
# =====================================================
def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 2]


def extract_keywords(text, n=TOP_N_KEYWORDS):
    tokens = tokenize(text)
    return pd.DataFrame(
        Counter(tokens).most_common(n),
        columns=["keyword", "freq"]
    )


def polarity(word):
    if word in POS_WORDS:
        return "Positive"
    if word in NEG_WORDS:
        return "Negative"
    return "Neutral"


def sentiment_from_keywords(df_kw):
    if df_kw.empty:
        return {"score": 0.0, "label": "Neutral", "pos": 0, "neg": 0, "neu": 0}

    pos = neg = neu = 0
    for _, r in df_kw.iterrows():
        if r["keyword"] in POS_WORDS:
            pos += r["freq"]
        elif r["keyword"] in NEG_WORDS:
            neg += r["freq"]
        else:
            neu += r["freq"]

    total = pos + neg + neu
    score = (pos - neg) / total if total else 0

    label = (
        "Positive" if score > POS_THRESHOLD
        else "Negative" if score < NEG_THRESHOLD
        else "Neutral"
    )

    return {
        "score": round(score, 2),
        "label": label,
        "pos": pos,
        "neg": neg,
        "neu": neu
    }


# =====================================================
# External Sentiment (REAL, EXTERNAL)
# =====================================================
def external_sentiment(text):
    """
    Uses TextBlob (external NLP library).
    Returns polarity between -1 and 1.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity


# =====================================================
# Color Scale
# =====================================================
COLOR_SCALE = alt.Scale(
    domain=["Positive", "Neutral", "Negative"],
    range=["#2ca02c", "#7f7f7f", "#d62728"]
)


# =====================================================
# Load Data
# =====================================================
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)


if "dataset" not in st.session_state:
    st.session_state.dataset = load_data()

df = st.session_state.dataset.copy()


# =====================================================
# Sidebar
# =====================================================
st.sidebar.title("Controls")

sentiment_mode = st.sidebar.radio(
    "Sentiment method",
    [
        "Keyword-based (Explainable)",
        "External NLP model (TextBlob)"
    ]
)

if df.empty:
    st.warning("No data available. Upload a CSV first.")
    st.stop()

event = st.sidebar.selectbox("Event", sorted(df["event_name"].unique()))
stakeholder = st.sidebar.selectbox("Stakeholder", sorted(df["stakeholder"].unique()))

filtered = df[(df["event_name"] == event) & (df["stakeholder"] == stakeholder)]
filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
filtered = filtered.sort_values("date")

row = filtered.iloc[0]


# =====================================================
# Main View
# =====================================================
st.title("Shifting Narratives")
st.caption("Comparing explainable and external sentiment analysis approaches.")

st.markdown(
    f"""
<div style='padding:16px;background:#f5f5f5;border-radius:10px;'>
<b>Document excerpt:</b><br>
{row["text"][:900]}{"..." if len(row["text"]) > 900 else ""}
</div>
""",
    unsafe_allow_html=True
)

# ---------- Sentiment Calculation ----------
if sentiment_mode == "External NLP model (TextBlob)":
    score = external_sentiment(row["text"])
    label = (
        "Positive" if score > POS_THRESHOLD
        else "Negative" if score < NEG_THRESHOLD
        else "Neutral"
    )

    st.markdown(
        f"<div style='font-size:18px;'>"
        f"<b>External sentiment score:</b> {score:.2f} ({label})"
        f"</div>",
        unsafe_allow_html=True
    )

    st.caption(
        "This score is produced by an external NLP model and is not directly explainable at word level."
    )

else:
    kw_df = extract_keywords(row["text"])
    kw_df["polarity"] = kw_df["keyword"].apply(polarity)
    stats = sentiment_from_keywords(kw_df)

    st.markdown(
        f"<div style='font-size:18px;'>"
        f"<b>Keyword-based score:</b> {stats['score']} ({stats['label']})"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("**Legend:** 🟢 Positive | ⚪ Neutral | 🔴 Negative")

    chart = (
        alt.Chart(kw_df.head(15))
        .mark_bar()
        .encode(
            x="freq:Q",
            y=alt.Y("keyword:N", sort="-x"),
            color=alt.Color("polarity:N", scale=COLOR_SCALE),
            tooltip=["keyword", "freq", "polarity"]
        )
        .properties(height=400, title="Keyword Contribution to Sentiment")
    )

    st.altair_chart(chart, use_container_width=True)

    st.markdown(
        """
**Score formula:**  
`(positive - negative) / total keyword frequency`
"""
    )


st.markdown("---")
if st.button("Done Exploring 🎉"):
    st.balloons()
