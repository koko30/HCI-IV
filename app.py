import re
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(page_title="Shifting Narratives", page_icon="🌍", layout="centered")


# =====================================================
# CONFIGURATION
# =====================================================
DATA_PATH = Path("data/events.csv")

REQUIRED_COLUMNS = ["event_name", "stakeholder", "text", "date"]
OPTIONAL_COLUMNS = ["entity", "source", "collection_method"]

TOP_N_KEYWORDS = 30
POS_THRESHOLD = 0.08
NEG_THRESHOLD = -0.08

NEGATION_WINDOW = 3
NEGATION_WORDS = {
    "not", "no", "never", "without", "hardly", "rarely", "none",
    "cannot", "can't", "dont", "don't", "doesnt", "doesn't"
}


# =====================================================
# SENTIMENT LEXICON (STARTER)
# =====================================================
POS_WORDS = {
    "progress", "success", "agreement", "cooperation", "growth", "innovation", "support",
    "stability", "unity", "hope", "benefit", "achievement", "positive", "safe",
    "responsible", "improve", "improved", "improving", "solution", "solutions", "peace"
}

NEG_WORDS = {
    "crisis", "failure", "conflict", "risk", "fear", "harm", "controversy", "discrimination",
    "violence", "loss", "problem", "negative", "collapse", "threat", "chaos", "scandal",
    "attack", "hate", "danger"
}

STOPWORDS = {
    "the","and","a","to","of","in","on","for","with","as","at","by","is","are","was","were",
    "be","been","being","it","this","that","from","or","but","we","they","you","i","he","she",
    "them","his","her","their","our","us","your","my","me","will","would","can","could","should",
    "may","might","about","into","over","after","before","more","most","some","any","so","such",
    "also","very","just","up","down","out","now","new","one","two","many","all"
}


# =====================================================
# TEXT PROCESSING
# =====================================================
def tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)  # keep apostrophes
    tokens = [t.strip("'") for t in text.split() if t.strip("'")]
    return tokens


def cleaned_tokens(text: str) -> list[str]:
    tokens = tokenize(text)
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def polarity(word: str) -> str:
    if word in POS_WORDS:
        return "Positive"
    if word in NEG_WORDS:
        return "Negative"
    return "Neutral"


def extract_keywords(text: str, n: int = TOP_N_KEYWORDS) -> pd.DataFrame:
    tokens = cleaned_tokens(text)
    freq = Counter(tokens).most_common(n)
    return pd.DataFrame(freq, columns=["keyword", "freq"])


def apply_negation_flip(token_list: list[str]) -> list[str]:
    """
    Return a polarity per token using a negation window:
    If a negation appears within the previous NEGATION_WINDOW tokens,
    flip Positive <-> Negative for sentiment words.
    """
    out = []
    for i, w in enumerate(token_list):
        base = polarity(w)

        negated = any(
            token_list[j] in NEGATION_WORDS
            for j in range(max(0, i - NEGATION_WINDOW), i)
        )

        if negated and base in {"Positive", "Negative"}:
            out.append("Negative" if base == "Positive" else "Positive")
        else:
            out.append(base)

    return out


def sentiment_stats_from_text(text: str) -> dict:
    """
    Keyword-frequency based sentiment:
    - Extract top keywords with counts
    - Expand into a frequency token stream
    - Apply negation flip
    - Count Positive/Negative/Neutral and compute score
    """
    kw_df = extract_keywords(text, TOP_N_KEYWORDS)

    if kw_df.empty:
        return {"score": 0.0, "label": "Neutral", "pos": 0, "neg": 0, "neu": 0, "coverage": 0.0, "kw_df": kw_df}

    expanded = []
    for _, r in kw_df.iterrows():
        expanded.extend([r["keyword"]] * int(r["freq"]))

    pols = apply_negation_flip(expanded)
    pos = int(sum(p == "Positive" for p in pols))
    neg = int(sum(p == "Negative" for p in pols))
    neu = int(sum(p == "Neutral" for p in pols))

    total = pos + neg + neu
    score = 0.0 if total == 0 else (pos - neg) / total

    if score > POS_THRESHOLD:
        label = "Positive"
    elif score < NEG_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    coverage = 0.0 if total == 0 else (pos + neg) / total

    kw_df["base_polarity"] = kw_df["keyword"].apply(polarity)

    return {"score": float(score), "label": label, "pos": pos, "neg": neg, "neu": neu, "coverage": float(coverage), "kw_df": kw_df}


# =====================================================
# VISUAL SETTINGS
# =====================================================
COLOR_SCALE = alt.Scale(
    domain=["Positive", "Neutral", "Negative"],
    range=["#2ca02c", "#7f7f7f", "#d62728"]
)


# =====================================================
# CSV TEMPLATE DOWNLOAD
# =====================================================
def template_csv_bytes() -> bytes:
    template = pd.DataFrame(
        {
            "event_name": ["Example Event"],
            "stakeholder": ["Government"],
            "entity": ["Ministry of Environment"],
            "source": ["https://example.com/article"],
            "collection_method": ["Official statement"],
            "date": ["2025-01-01"],
            "text": ["Paste your article text here..."],
        }
    )
    return template.to_csv(index=False).encode("utf-8")


# =====================================================
# LOAD INITIAL DATA
# =====================================================
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)


if "dataset" not in st.session_state:
    st.session_state.dataset = load_data()

df = st.session_state.dataset.copy()


# =====================================================
# SIDEBAR: DATA INPUT
# =====================================================
st.sidebar.title("Data Input")

input_mode = st.sidebar.radio("Input method", ["Upload CSV", "Paste Article"], index=0)

st.sidebar.download_button(
    label="Download CSV Template",
    data=template_csv_bytes(),
    file_name="events_template.csv",
    mime="text/csv"
)

if input_mode == "Upload CSV":
    st.sidebar.subheader("Upload Dataset (CSV)")

    with st.sidebar.expander("📘 CSV Instructions"):
        st.markdown(
            """
**Each row = one document/article**

**Required columns**
- `event_name`
- `stakeholder`
- `text`
- `date` (YYYY-MM-DD)

**Optional columns**
- `entity`
- `source`
- `collection_method`
"""
        )

    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df_new = pd.read_csv(uploaded_file)

            if df_new.empty:
                st.sidebar.error("Upload failed: The CSV has no rows.")
            else:
                missing = [c for c in REQUIRED_COLUMNS if c not in df_new.columns]
                if missing:
                    st.sidebar.error(f"Upload failed: Missing required columns: {missing}")
                else:
                    df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce")
                    if df_new["date"].isna().any():
                        st.sidebar.error("Date format error: Use YYYY-MM-DD in the date column.")
                    else:
                        for c in OPTIONAL_COLUMNS:
                            if c not in df_new.columns:
                                df_new[c] = ""
                        st.session_state.dataset = df_new.copy()
                        df = st.session_state.dataset.copy()
                        st.sidebar.success("CSV loaded successfully.")
        except Exception:
            st.sidebar.error("Upload failed: Could not read CSV. Check commas/quotes.")

else:
    st.sidebar.subheader("Add Article")
    event_name = st.sidebar.text_input("Event name", value="New Event")
    stakeholder = st.sidebar.selectbox("Stakeholder", ["Government", "Media", "Public"])
    entity = st.sidebar.text_input("Entity (optional)", value="")
    source = st.sidebar.text_input("Source (optional)", value="")
    collection_method = st.sidebar.text_input("Collection method (optional)", value="Manual paste")
    date = st.sidebar.date_input("Date")
    text = st.sidebar.text_area("Paste article text", height=200)

    if st.sidebar.button("Add Article"):
        if not text.strip():
            st.sidebar.error("Please paste an article text before adding.")
        else:
            new_row = {
                "event_name": event_name.strip(),
                "stakeholder": stakeholder,
                "entity": entity.strip(),
                "source": source.strip(),
                "collection_method": collection_method.strip(),
                "date": str(date),
                "text": text.strip(),
            }
            st.session_state.dataset = pd.concat(
                [st.session_state.dataset, pd.DataFrame([new_row])],
                ignore_index=True
            )
            df = st.session_state.dataset.copy()
            st.sidebar.success("Article added.")


# =====================================================
# SIDEBAR: VIEW CONTROLS
# =====================================================
st.sidebar.title("View Controls")

if df.empty:
    st.warning("No data available yet. Upload a CSV or paste an article.")
    st.stop()

for c in REQUIRED_COLUMNS:
    if c not in df.columns:
        df[c] = ""
for c in OPTIONAL_COLUMNS:
    if c not in df.columns:
        df[c] = ""

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()
df["text"] = df["text"].astype(str)

event_list = sorted([e for e in df["event_name"].dropna().unique() if str(e).strip()])
stake_list = sorted([s for s in df["stakeholder"].dropna().unique() if str(s).strip()])

if not event_list or not stake_list:
    st.warning("Dataset does not contain valid event_name/stakeholder values.")
    st.stop()

selected_event = st.sidebar.selectbox("Select Event", event_list)
selected_stakeholder = st.sidebar.selectbox("Select Stakeholder", stake_list)

event_df = df[df["event_name"] == selected_event].copy()
stake_df = event_df[event_df["stakeholder"] == selected_stakeholder].copy()

entities = ["All"] + sorted([e for e in stake_df["entity"].dropna().unique() if str(e).strip()])
selected_entity = st.sidebar.selectbox("Select Entity", entities)

if selected_entity != "All":
    stake_df = stake_df[stake_df["entity"] == selected_entity].copy()

if stake_df.empty:
    st.warning("No documents found for the selected filters.")
    st.stop()

stake_df = stake_df.sort_values("date")
doc_indices = stake_df.index.tolist()
doc_label_map = {
    idx: f"{stake_df.loc[idx,'date'].date()} — {str(stake_df.loc[idx,'entity']).strip() or 'No entity'}"
    for idx in doc_indices
}
selected_doc_idx = st.sidebar.selectbox(
    "Select Document Example",
    options=doc_indices,
    format_func=lambda x: doc_label_map.get(x, str(x))
)

row = stake_df.loc[selected_doc_idx]


# =====================================================
# MAIN DASHBOARD
# =====================================================
st.title("Shifting Narratives")
st.caption("Keyword-frequency sentiment with negation handling (e.g., 'not good').")

with st.expander("What does 'frequency' mean here?"):
    st.markdown(
        """
**Frequency** means how many times a word appears in the text after cleaning.  
Example: if **crisis** appears 7 times, its frequency is **7**.

We extract the most frequent words, then count how many of those frequent words are:
🟢 Positive, ⚪ Neutral, 🔴 Negative (including negation flipping).
"""
    )

st.markdown("### Dataset Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Events", len(df["event_name"].unique()))
c2.metric("Documents", len(df))
c3.metric("Stakeholders", len(df["stakeholder"].unique()))
c4.metric("Date range", f"{df['date'].min().date()} → {df['date'].max().date()}")

st.markdown("---")
st.subheader(f"{selected_stakeholder} Perspective — {selected_event}")

doc_stats = sentiment_stats_from_text(row["text"])
doc_kw = doc_stats["kw_df"].copy()

sent_color = "#2ca02c" if doc_stats["label"] == "Positive" else "#d62728" if doc_stats["label"] == "Negative" else "#7f7f7f"

st.markdown(
    f"""
<div style='padding:18px;border-radius:10px;background:#f5f5f5;font-size:18px;line-height:1.6;'>
<b>Document text (excerpt):</b><br>
{row['text'][:900]}{"..." if len(row['text']) > 900 else ""}
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    f"<div style='font-size:18px;color:{sent_color};margin-top:10px;font-weight:bold;'>"
    f"Sentiment score: {doc_stats['score']:.2f} ({doc_stats['label']})</div>",
    unsafe_allow_html=True
)

st.markdown("**Legend:** 🟢 Positive   |   ⚪ Neutral   |   🔴 Negative")

# =====================================================
# WHY THIS SCORE?
# =====================================================
st.markdown("---")
st.markdown("## Why this sentiment score?")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Positive freq", doc_stats["pos"])
m2.metric("Negative freq", doc_stats["neg"])
m3.metric("Neutral freq", doc_stats["neu"])
m4.metric("Coverage", f"{doc_stats['coverage']*100:.0f}%")

st.caption("Coverage shows how much of the counted frequent words are emotional (positive/negative).")

top_kw = doc_kw.sort_values("freq", ascending=False).head(15).copy()
top_kw["polarity"] = top_kw["keyword"].apply(polarity)

keyword_bar = (
    alt.Chart(top_kw)
    .mark_bar()
    .encode(
        x=alt.X("freq:Q", title="Frequency"),
        y=alt.Y("keyword:N", sort="-x", title="Keyword"),
        color=alt.Color("polarity:N", scale=COLOR_SCALE),
        tooltip=["keyword", "freq", "polarity"]
    )
    .properties(height=420, title="Top Keywords by Frequency (Base Polarity)")
)
st.altair_chart(keyword_bar, use_container_width=True)

st.markdown(
    """
**Score formula:**  
`score = (positive_freq - negative_freq) / (positive_freq + negative_freq + neutral_freq)`  

**Negation rule:** if a negation word (not/no/never/without) appears shortly before a sentiment word,
we flip the polarity (example: “not good” → negative).
"""
)

# =====================================================
# EVENT PIE (DONUT) CHART WITH ALTAIR (NO MATPLOTLIB)
# =====================================================
st.markdown("---")
st.markdown("## Selected Event: Effective Words (Percent)")

event_pos = 0
event_neg = 0
event_neu = 0

for t in event_df["text"].astype(str).tolist():
    s = sentiment_stats_from_text(t)
    event_pos += s["pos"]
    event_neg += s["neg"]
    event_neu += s["neu"]

total = event_pos + event_neg + event_neu

if total == 0:
    st.info("No effective keywords found for this event yet.")
else:
    pie_df = pd.DataFrame(
        {
            "category": ["Positive", "Neutral", "Negative"],
            "count": [event_pos, event_neu, event_neg],
        }
    )
    pie_df["percent"] = pie_df["count"] / pie_df["count"].sum()

    donut = (
        alt.Chart(pie_df)
        .mark_arc(innerRadius=60)
        .encode(
            theta=alt.Theta("count:Q", stack=True),
            color=alt.Color("category:N", scale=COLOR_SCALE, title=None),
            tooltip=[
                "category:N",
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("percent:Q", format=".1%", title="Percent"),
            ],
        )
        .properties(height=320)
    )

    labels = (
        alt.Chart(pie_df)
        .mark_text(radius=130, size=14)
        .encode(
            theta=alt.Theta("count:Q", stack=True),
            text=alt.Text("percent:Q", format=".0%"),
            color=alt.value("black"),
        )
    )

    st.altair_chart(donut + labels, use_container_width=True)

    st.caption("Percentages are based on the effective frequent words found (with negation flipping).")

# =====================================================
# STAKEHOLDER OVERVIEW
# =====================================================
st.markdown("---")
st.markdown("## Stakeholder Overview (Selected Event)")

def score_for_text(text: str) -> float:
    return sentiment_stats_from_text(text)["score"]

overview = (
    event_df.assign(score=event_df["text"].astype(str).apply(score_for_text))
    .groupby("stakeholder", as_index=False)["score"]
    .mean()
    .rename(columns={"score": "avg_score"})
)

overview["category"] = overview["avg_score"].apply(
    lambda s: "Positive" if s > POS_THRESHOLD else "Negative" if s < NEG_THRESHOLD else "Neutral"
)

overview_chart = (
    alt.Chart(overview)
    .mark_bar()
    .encode(
        x=alt.X("stakeholder:N", title="Stakeholder"),
        y=alt.Y("avg_score:Q", title="Average sentiment", scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color("category:N", scale=COLOR_SCALE, title=None),
        tooltip=["stakeholder", alt.Tooltip("avg_score:Q", format=".2f")]
    )
    .properties(height=260)
)
st.altair_chart(overview_chart, use_container_width=True)

# =====================================================
# TIMELINE (SELECTED EVENT)
# =====================================================
st.markdown("---")
st.markdown("## Timeline: Sentiment Over Time (Selected Event)")

timeline_df = event_df.copy()
timeline_df["score"] = timeline_df["text"].astype(str).apply(score_for_text)

timeline_chart = (
    alt.Chart(timeline_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y", labelAngle=-35)),
        y=alt.Y("score:Q", title="Sentiment Score", scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color("stakeholder:N", title="Stakeholder"),
        tooltip=["event_name", "stakeholder", "entity", alt.Tooltip("score:Q", format=".2f"), "date"]
    )
    .properties(height=260)
)
st.altair_chart(timeline_chart, use_container_width=True)

st.markdown("---")
if st.button("Done Exploring 🎉"):
    st.balloons()
