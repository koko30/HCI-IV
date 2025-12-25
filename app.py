import re
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


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
# Sentiment Lexicon (Starter)
# =====================================================
POS_WORDS = {
"good","great","excellent","positive","success","successful","progress","growth","improve","improved",
"improving","improvement","benefit","beneficial","advantage","advantageous","achievement","achieve",
"achieved","innovation","innovative","support","supportive","stability","stable","unity","hope",
"hopeful","optimistic","confidence","confident","trust","trusted","reliable","reliability","secure",
"security","safe","safety","peace","peaceful","cooperation","collaboration","agreement","consensus",
"fair","fairness","balanced","responsible","responsibility","ethical","transparent","clarity",
"efficient","efficiency","effective","effectiveness","productive","productivity","strong","strength",
"resilient","resilience","sustainable","sustainability","inclusive","inclusion","empower","empowered",
"empowering","successfully","prosper","prosperity","advance","advancing","advancement","achievement",
"reward","rewarding","encouraging","encouraged","encourage","positive","constructive","helpful",
"useful","valuable","value","worth","worthy","commendable","commend","celebrate","celebration",
"proud","pride","respect","respectful","coherent","consistent","credible","legitimate","valid",
"verified","accurate","accuracy","optimism","calm","confidence","clarified","impressive","robust",
"progressive","dynamic","adaptive","responsive","friendly","cooperative","supporting","guiding",
"motivating","motivated","motivational","uplifting","reassuring","securely","stabilize","stabilized",
"strengthen","strengthened","strengthening","upgrade","upgraded","upgrading","win","winning",
"victory","achiever","highquality","best","better","improved","improvement","progression"
}


NEG_WORDS = {
"bad","poor","negative","failure","fail","failed","failing","risk","risky","danger","dangerous",
"threat","threaten","threatening","crisis","conflict","controversy","controversial","harm","harmful",
"damage","damaging","loss","decline","declining","collapse","collapsed","weak","weakness","unstable",
"instability","unsafe","insecure","fear","fearful","panic","anxiety","angry","anger","frustrated",
"frustration","disappointed","disappointment","problem","problems","issue","issues","chaos","chaotic",
"scandal","corruption","corrupt","bias","biased","unfair","injustice","inequality","discrimination",
"violence","violent","abuse","abusive","attack","attacked","attacking","hate","hatred","hostile",
"hostility","toxic","misleading","false","falsehood","lie","lying","deception","deceptive","fraud",
"fraudulent","error","errors","mistake","mistakes","delay","delayed","delay","breakdown","collapse",
"crash","crisis","deficit","shortage","failure","ineffective","inefficient","mismanagement",
"mismanaged","poorly","weakly","unstable","uncertain","uncertainty","doubt","doubtful","skeptical",
"skepticism","threatened","collapse","harmfully","destructive","destruction","chaos","instability",
"opposition","protest","outrage","criticize","criticism","criticized","blame","accusation","accused"
}


STOPWORDS = {
    "the","and","a","to","of","in","on","for","with","as","at","by",
    "is","are","was","were","be","been","being","it","this","that",
    "from","or","but","not","we","they","you","i","he","she","them",
    "his","her","their","our","us","your","my","me","will","would",
    "can","could","should","may","might","about","into","over","after",
    "before","more","most","some","any","no","yes","if","then","than",
    "so","such","also","very","just","up","down","out","now","new",
    "one","two","many","all"
}

# =====================================================
# Text Processing Helpers
# =====================================================
def tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return [t for t in text.split() if len(t) > 2 and t not in STOPWORDS]

def extract_keywords(text: str, n: int = TOP_N_KEYWORDS) -> pd.DataFrame:
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
    if df_kw.empty:
        return {"score": 0.0, "label": "Neutral", "pos": 0, "neg": 0, "neu": 0, "coverage": 0.0}

    df = df_kw.copy()
    df["polarity"] = df["keyword"].apply(polarity)

    pos = int(df[df["polarity"] == "Positive"]["freq"].sum())
    neg = int(df[df["polarity"] == "Negative"]["freq"].sum())
    neu = int(df[df["polarity"] == "Neutral"]["freq"].sum())

    total = pos + neg + neu
    score = 0.0 if total == 0 else (pos - neg) / total

    if score > POS_THRESHOLD:
        label = "Positive"
    elif score < NEG_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    coverage = 0.0 if total == 0 else (pos + neg) / total

    return {"score": float(score), "label": label, "pos": pos, "neg": neg, "neu": neu, "coverage": float(coverage)}

# =====================================================
# Altair Color Scale
# =====================================================
COLOR_SCALE = alt.Scale(
    domain=["Positive", "Neutral", "Negative"],
    range=["#2ca02c", "#7f7f7f", "#d62728"]
)

# =====================================================
# Load Initial Data
# =====================================================
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    # Fallback empty dataset with columns
    return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

if "dataset" not in st.session_state:
    st.session_state.dataset = load_data()

df = st.session_state.dataset.copy()

# =====================================================
# Sidebar: Data Input
# =====================================================
st.sidebar.title("Data Input")

input_mode = st.sidebar.radio(
    "Input method",
    ["Upload CSV", "Paste Article"]
)

# ---------- Upload CSV Instructions Button ----------
if input_mode == "Upload CSV":
    st.sidebar.subheader("Upload Dataset (CSV)")

    with st.sidebar.expander("📘 CSV Instructions"):
        st.markdown(
            """
**What should this CSV contain?**

Each row should represent **one article or document**.

### Required columns
- **event_name**: Name of the event (used for grouping)
- **stakeholder**: Perspective (Government / Media / Public, etc.)
- **text**: Full article text or cleaned summary
- **date**: Publication date in format **YYYY-MM-DD**

### Optional columns
- **entity**: Specific source (BBC, Ministry, Twitter, etc.)
- **source**: URL or platform name
- **collection_method**: How the text was collected (news, statement, social post)

If your CSV follows this structure, the app will work automatically.
"""
        )

    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_new = pd.read_csv(uploaded_file)

            if df_new.empty:
                st.sidebar.error("Upload failed: The CSV file contains no rows.")
            else:
                missing = [c for c in REQUIRED_COLUMNS if c not in df_new.columns]
                if missing:
                    st.sidebar.error(f"Upload failed: Missing required columns: {missing}")
                else:
                    df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce")
                    if df_new["date"].isna().any():
                        st.sidebar.error("Date format error: Please use YYYY-MM-DD in the date column.")
                    else:
                        # Ensure optional columns exist
                        for c in OPTIONAL_COLUMNS:
                            if c not in df_new.columns:
                                df_new[c] = ""
                        st.session_state.dataset = df_new.copy()
                        df = st.session_state.dataset.copy()
                        st.sidebar.success("CSV loaded successfully. Dataset is ready.")

        except Exception:
            st.sidebar.error("Upload failed: Could not read the CSV file. Please check formatting.")

# ---------- Paste Article Mode ----------
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
# Sidebar: View Controls (Moved here)
# =====================================================
st.sidebar.title("View Controls")

if df.empty:
    st.warning("No data available yet. Upload a CSV or paste an article.")
    st.stop()

# Ensure required columns exist
for c in REQUIRED_COLUMNS:
    if c not in df.columns:
        df[c] = ""

for c in OPTIONAL_COLUMNS:
    if c not in df.columns:
        df[c] = ""

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()
df["text"] = df["text"].astype(str)

# Event selection
event_list = sorted([e for e in df["event_name"].dropna().unique() if str(e).strip() != ""])
if not event_list:
    st.warning("No valid events found in dataset.")
    st.stop()

selected_event = st.sidebar.selectbox("Select Event", event_list)

# Stakeholder selection
stake_list = sorted([s for s in df["stakeholder"].dropna().unique() if str(s).strip() != ""])
selected_stakeholder = st.sidebar.selectbox("Select Stakeholder", stake_list)

# Filter
event_df = df[df["event_name"] == selected_event].copy()
stake_df = event_df[event_df["stakeholder"] == selected_stakeholder].copy()

# Entity selection
entities = ["All"] + sorted([e for e in stake_df["entity"].dropna().unique() if str(e).strip() != ""])
selected_entity = st.sidebar.selectbox("Select Entity", entities)

if selected_entity != "All":
    stake_df = stake_df[stake_df["entity"] == selected_entity].copy()

if stake_df.empty:
    st.warning("No documents found for the selected filters.")
    st.stop()

# Document selection (example)
stake_df = stake_df.sort_values("date")
doc_indices = stake_df.index.tolist()
doc_label_map = {
    idx: f"{stake_df.loc[idx,'date'].date()} — {str(stake_df.loc[idx,'entity'])[:35] if str(stake_df.loc[idx,'entity']).strip() else 'No entity'}"
    for idx in doc_indices
}

selected_doc_idx = st.sidebar.selectbox(
    "Select Document Example",
    options=doc_indices,
    format_func=lambda x: doc_label_map.get(x, str(x))
)

row = stake_df.loc[selected_doc_idx]

# =====================================================
# Main Dashboard
# =====================================================
st.title("Shifting Narratives")
st.caption("Explainable sentiment scoring using keyword frequency and a sentiment dictionary.")

# Dataset summary
st.markdown("### Dataset Summary")
colA, colB, colC, colD = st.columns(4)
colA.metric("Events", len(df["event_name"].unique()))
colB.metric("Documents", len(df))
colC.metric("Stakeholders", len(df["stakeholder"].unique()))
colD.metric("Date range", f"{df['date'].min().date()} → {df['date'].max().date()}")

# Show collected sources and methods (if present)
with st.expander("See data collection details"):
    st.write("**Sources:**")
    sources = sorted([s for s in df["source"].dropna().unique() if str(s).strip() != ""])
    st.write(sources if sources else "No sources provided.")

    st.write("**Collection methods:**")
    methods = sorted([m for m in df["collection_method"].dropna().unique() if str(m).strip() != ""])
    st.write(methods if methods else "No collection methods provided.")

st.markdown("---")

# Selected view header
st.subheader(f"{selected_stakeholder} Perspective — {selected_event}")

# Compute sentiment + keywords for selected doc
kw_df = extract_keywords(row["text"], TOP_N_KEYWORDS)
kw_df["polarity"] = kw_df["keyword"].apply(polarity)
stats = sentiment_from_keywords(kw_df)

sent_color = "#2ca02c" if stats["label"] == "Positive" else "#d62728" if stats["label"] == "Negative" else "#7f7f7f"

# Show document text excerpt
st.markdown(
    f"""
<div style='padding:18px;border-radius:10px;background:#f5f5f5;font-size:18px;line-height:1.6;'>
<b>Document text (excerpt):</b><br>
{row['text'][:900]}{"..." if len(row['text']) > 900 else ""}
</div>
""",
    unsafe_allow_html=True
)

# Sentiment result
st.markdown(
    f"<div style='font-size:18px;color:{sent_color};margin-top:10px;font-weight:bold;'>"
    f"Sentiment score: {stats['score']:.2f} ({stats['label']})</div>",
    unsafe_allow_html=True
)

# Legend placed near the results
st.markdown("**Legend:** 🟢 Positive   |   ⚪ Neutral   |   🔴 Negative")

st.markdown("---")

# =====================================================
# Explanation: Why this score?
# =====================================================
st.markdown("## Why this sentiment score?")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Positive freq", stats["pos"])
m2.metric("Negative freq", stats["neg"])
m3.metric("Neutral freq", stats["neu"])
m4.metric("Coverage", f"{stats['coverage']*100:.0f}%")

st.caption("Coverage shows how much of the top keywords are emotional (positive/negative) instead of neutral.")

top_kw = kw_df.sort_values("freq", ascending=False).head(15)

keyword_bar = (
    alt.Chart(top_kw)
    .mark_bar()
    .encode(
        x=alt.X("freq:Q", title="Frequency"),
        y=alt.Y("keyword:N", sort="-x", title="Keyword"),
        color=alt.Color("polarity:N", scale=COLOR_SCALE),
        tooltip=["keyword", "freq", "polarity"]
    )
    .properties(height=420, title="Top Keywords by Frequency and Polarity")
)

st.altair_chart(keyword_bar, use_container_width=True)

st.markdown(
    """
**Score formula:**  
`score = (positive_freq - negative_freq) / (positive_freq + negative_freq + neutral_freq)`  
"""
)

st.markdown("---")

# =====================================================
# Stakeholder Overview (Selected Event)
# =====================================================
st.markdown("## Stakeholder Overview (Selected Event)")

def avg_sentiment(texts: pd.Series) -> float:
    scores = []
    for t in texts.astype(str).tolist():
        kw = extract_keywords(t, TOP_N_KEYWORDS)
        scores.append(sentiment_from_keywords(kw)["score"])
    return sum(scores) / len(scores) if scores else 0.0

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
        x=alt.X("stakeholder:N", title="Stakeholder"),
        y=alt.Y("avg_score:Q", title="Average sentiment", scale=alt.Scale(domain=[-1, 1])),
        color=alt.Color("category:N", scale=COLOR_SCALE),
        tooltip=["stakeholder", alt.Tooltip("avg_score:Q", format=".2f")]
    )
    .properties(height=260)
)

st.altair_chart(overview_chart, use_container_width=True)

st.markdown("---")

# =====================================================
# Timeline (Selected Event Only)
# =====================================================
st.markdown("## Timeline: Sentiment Over Time (Selected Event)")

# Compute sentiment score per row for timeline
timeline_df = event_df.copy()
scores = []
for t in timeline_df["text"].astype(str).tolist():
    kw = extract_keywords(t, TOP_N_KEYWORDS)
    scores.append(sentiment_from_keywords(kw)["score"])

timeline_df["score"] = scores

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
