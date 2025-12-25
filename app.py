import re
from collections import Counter
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
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

# Sentiment thresholds for labeling an overall score
POS_THRESHOLD = 0.08
NEG_THRESHOLD = -0.08

# Negation handling: flip polarity for sentiment words within this window after a negation token
NEGATION_WINDOW = 3
NEGATION_WORDS = {"not", "no", "never", "without", "hardly", "rarely", "none", "cannot", "can't", "don't", "doesn't"}


# =====================================================
# SENTIMENT LEXICON (STARTER)
# =====================================================
# Feel free to extend these lists for your topic/domain.
POS_WORDS = {
    "progress", "success", "agreement", "cooperation", "growth", "innovation", "support",
    "stability", "unity", "hope", "benefit", "achievement", "positive", "safe", "responsible",
    "improve", "improved", "improving", "solution", "solutions", "peace"
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
    """TURN RAW TEXT INTO NORMALIZED TOKENS."""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)  # Keep apostrophes for contractions like don't
    tokens = [t.strip("'") for t in text.split() if t.strip("'")]
    return tokens


def cleaned_tokens(text: str) -> list[str]:
    """REMOVE STOPWORDS AND SHORT TOKENS, KEEP USEFUL WORDS."""
    tokens = tokenize(text)
    out = []
    for t in tokens:
        if len(t) <= 2:
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    return out


def polarity(word: str) -> str:
    """RETURN BASE POLARITY FROM THE LEXICON."""
    if word in POS_WORDS:
        return "Positive"
    if word in NEG_WORDS:
        return "Negative"
    return "Neutral"


def apply_negation_flip(tokens: list[str]) -> list[tuple[str, str]]:
    """
    CLASSIFY TOKENS WITH A NEGATION WINDOW.
    If a negation word appears, sentiment words within the next NEGATION_WINDOW tokens get flipped.
    Neutral words stay neutral.
    """
    results = []
    n = len(tokens)

    for i, w in enumerate(tokens):
        base = polarity(w)

        # Detect if there is a negation within the previous window
        negated = False
        start = max(0, i - NEGATION_WINDOW)
        for j in range(start, i):
            if tokens[j] in NEGATION_WORDS:
                negated = True
                break

        if negated and base in {"Positive", "Negative"}:
            flipped = "Negative" if base == "Positive" else "Positive"
            results.append((w, flipped))
        else:
            results.append((w, base))

    return results


def extract_keywords(text: str, n: int = TOP_N_KEYWORDS) -> pd.DataFrame:
    """EXTRACT TOP-N KEYWORDS BY FREQUENCY (AFTER CLEANING)."""
    tokens = cleaned_tokens(text)
    freq = Counter(tokens).most_common(n)
    return pd.DataFrame(freq, columns=["keyword", "freq"])


def sentiment_stats_from_text(text: str) -> dict:
    """
    COMPUTE SENTIMENT USING KEYWORD FREQUENCY + NEGATION FLIP.
    The counts are based on the TOP keywords by frequency.
    """
    kw = extract_keywords(text, TOP_N_KEYWORDS)

    if kw.empty:
        return {"score": 0.0, "label": "Neutral", "pos": 0, "neg": 0, "neu": 0, "coverage": 0.0, "kw_df": kw}

    # Build a token stream limited to top keywords, expanded by frequency
    # This keeps the method consistent with the "keyword frequency" idea.
    expanded = []
    for _, r in kw.iterrows():
        expanded.extend([r["keyword"]] * int(r["freq"]))

    classified = apply_negation_flip(expanded)
    pos = sum(1 for _, p in classified if p == "Positive")
    neg = sum(1 for _, p in classified if p == "Negative")
    neu = sum(1 for _, p in classified if p == "Neutral")

    total = pos + neg + neu
    score = 0.0 if total == 0 else (pos - neg) / total

    if score > POS_THRESHOLD:
        label = "Positive"
    elif score < NEG_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    coverage = 0.0 if total == 0 else (pos + neg) / total

    # Attach polarity for visualization (based on single word + negation context is shown in stats, not per keyword)
    kw["base_polarity"] = kw["keyword"].apply(polarity)

    return {"score": float(score), "label": label, "pos": int(pos), "neg": int(neg), "neu": int(neu), "coverage": float(coverage), "kw_df": kw}


# =====================================================
# VISUAL SETTINGS
# =====================================================
COLOR_SCALE = alt.Scale(domain=["Positive", "Neutral", "Negative"], range=["#2ca02c", "#7f7f7f", "#d62728"])


# =====================================================
# TEMPLATE CSV CONTENT (FOR DOWNLOAD)
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
# SIDEBAR: DATA INPUT + INSTRUCTIONS
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
**What should this CSV contain?**  
Each row represents **one article/document**.

**Required columns**
- `event_name` (grouping)
- `stakeholder` (Government / Media / Public or your own categories)
- `text` (article text or summary)
- `date` (YYYY-MM-DD)

**Optional columns**
- `entity` (BBC, CNN, Ministry, etc.)
- `source` (URL or platform)
- `collection_method` (news article, official statement, social post)
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
                        st.sidebar.error("Date format error: Please use YYYY-MM-DD in the date column.")
                    else:
                        for c in OPTIONAL_COLUMNS:
                            if c not in df_new.columns:
                                df_new[c] = ""
                        st.session_state.dataset = df_new.copy()
                        df = st.session_state.dataset.copy()
                        st.sidebar.success("CSV loaded successfully. Dataset is ready.")
        except Exception:
            st.sidebar.error("Upload failed: Could not read the CSV. Check commas and quotes.")

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
            st.session_state.dataset = pd.concat([st.session_state.dataset, pd.DataFrame([new_row])], ignore_index=True)
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
selected_doc_idx = st.sidebar.selectbox("Select Document Example", options=doc_indices, format_func=lambda x: doc_label_map.get(x, str(x)))

row = stake_df.loc[selected_doc_idx]


# =====================================================
# MAIN DASHBOARD
# =====================================================
st.title("Shifting Narratives")
st.caption("Explainable sentiment scoring using keyword frequency and a negation rule (e.g., 'not good').")

# -----------------------------------------------------
# Explain "Frequency score" in simple words
# -----------------------------------------------------
with st.expander("What does 'frequency' mean here?"):
    st.markdown(
        """
**Frequency** means: *how many times a word appears in the text (after cleaning).*  
Example: if the word **"crisis"** appears 7 times, its frequency is **7**.

We use frequency because repeated words usually represent the main focus of the text.
Then we count how many frequent words are **Positive**, **Negative**, or **Neutral**.
"""
    )

# -----------------------------------------------------
# Dataset summary
# -----------------------------------------------------
st.markdown("### Dataset Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Events", len(df["event_name"].unique()))
c2.metric("Documents", len(df))
c3.metric("Stakeholders", len(df["stakeholder"].unique()))
c4.metric("Date range", f"{df['date'].min().date()} → {df['date'].max().date()}")

st.markdown("---")
st.subheader(f"{selected_stakeholder} Perspective — {selected_event}")

# -----------------------------------------------------
# Selected document stats
# -----------------------------------------------------
doc_stats = sentiment_stats_from_text(row["text"])
doc_kw = doc_stats["kw_df"]

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

st.markdown("---")
st.markdown("## Why this sentiment score?")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Positive freq", doc_stats["pos"])
m2.metric("Negative freq", doc_stats["neg"])
m3.metric("Neutral freq", doc_stats["neu"])
m4.metric("Coverage", f"{doc_stats['coverage']*100:.0f}%")

st.caption("Coverage shows how much of the counted frequent words are emotional (positive/negative).")

# -----------------------------------------------------
# Keyword bar chart (clean replacement for messy word cloud)
# -----------------------------------------------------
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

Negation rule: If a negation word (like **not**, **no**, **never**) appears shortly before a sentiment word, the polarity is flipped.
"""
)

# =====================================================
# PIE CHART: SELECTED EVENT (PERCENTAGE OF EFFECTIVE WORDS)
# =====================================================
st.markdown("---")
st.markdown("## Selected Event: Effective Word Polarity (Percent)")

# Aggregate sentiment counts across all documents in the selected event
event_pos = 0
event_neg = 0
event_neu = 0

for t in event_df["text"].astype(str).tolist():
    s = sentiment_stats_from_text(t)
    event_pos += s["pos"]
    event_neg += s["neg"]
    event_neu += s["neu"]

total_effective = event_pos + event_neg + event_neu

if total_effective == 0:
    st.info("No effective keywords found for this event yet.")
else:
    labels = ["Positive", "Neutral", "Negative"]
    values = [event_pos, event_neu, event_neg]

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    st.caption(
        "This pie chart shows the percentage of effective frequent words for the selected event, "
        "classified as Positive, Neutral, or Negative (with negation handling)."
    )

# =====================================================
# OVERVIEW: AVERAGE SCORE BY STAKEHOLDER (SELECTED EVENT)
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
        color=alt.Color("category:N", scale=COLOR_SCALE),
        tooltip=["stakeholder", alt.Tooltip("avg_score:Q", format=".2f")]
    )
    .properties(height=260)
)
st.altair_chart(overview_chart, use_container_width=True)

# =====================================================
# TIMELINE: SELECTED EVENT ONLY
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
