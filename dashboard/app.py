import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Persona Responses Viewer",
    page_icon=":speech_balloon:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Force light theme and Inspect-like styling
st.markdown("""
<style>
    /* Force light mode */
    .stApp {
        background-color: #f0f2f5 !important;
    }

    .main .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 8px;
        max-width: 1400px;
    }

    .main-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #1a1a1a !important;
    }

    .submission-content {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
        white-space: pre-wrap;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #1a1a1a !important;
    }

    /* Fix expander text color */
    .stExpander {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }

    .stExpander summary {
        color: #1a1a1a !important;
    }

    .stExpander summary span {
        color: #1a1a1a !important;
    }

    .stExpander p, .stExpander span {
        color: #1a1a1a !important;
    }

    /* Make all text dark */
    p, span, label, div {
        color: #1a1a1a;
    }

    /* Stats metrics */
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
    }

    [data-testid="stMetricLabel"] {
        color: #555555 !important;
    }

    /* Response box styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        border-bottom: none;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f0;
        border: 1px solid #ddd;
        border-bottom: none;
        border-radius: 6px 6px 0 0;
        color: #1a1a1a !important;
        padding: 10px 20px;
        margin-right: 4px;
        font-weight: normal;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #ffffff;
        position: relative;
        z-index: 1;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 0 6px 6px 6px;
        padding: 1rem;
        margin-top: -1px;
    }

    /* Input fields */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #ced4da;
    }

    .stTextInput label {
        color: #1a1a1a !important;
    }

    .stSelectbox label {
        color: #1a1a1a !important;
    }

    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }

    /* Caption */
    .stCaption, small {
        color: #555555 !important;
    }

    /* Model response box */
    .model-response {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 1rem;
        height: 100%;
        white-space: pre-wrap;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #1a1a1a;
    }

    .model-header {
        font-weight: 600;
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #eee;
    }

    .not-available {
        color: #999;
        font-style: italic;
        padding: 1rem;
        text-align: center;
        background-color: #f5f5f5;
        border-radius: 6px;
    }

    /* Model toggle buttons */
    .stButton > button[kind="primary"] {
        background-color: #ff9800 !important;
        border-color: #ff9800 !important;
        color: white !important;
    }

    .stButton > button[kind="secondary"] {
        background-color: #e0e0e0 !important;
        border-color: #ccc !important;
        color: #666 !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: #ffcc80 !important;
        border-color: #ff9800 !important;
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Open Character Training: Advice Subreddit Evals</div>', unsafe_allow_html=True)

# File selector - toggle buttons
results_dir = Path("data/results")
csv_files = sorted(results_dir.glob("*.csv"), reverse=True) if results_dir.exists() else []

if not csv_files:
    st.warning("No CSV files found in data/results/")
    st.stop()

# Cache model names from all files at once
@st.cache_data
def get_all_model_names(file_paths):
    names = {}
    for fp in file_paths:
        df = pd.read_csv(fp, nrows=1, usecols=lambda c: c == 'model')
        names[str(fp)] = df['model'].iloc[0] if 'model' in df.columns else Path(fp).stem
    return names

model_names_map = get_all_model_names(tuple(csv_files))

# Initialize session state for file selection
if "selected_files" not in st.session_state:
    st.session_state.selected_files = {str(csv_files[0])} if csv_files else set()

def toggle_file(file_key):
    if file_key in st.session_state.selected_files:
        st.session_state.selected_files.discard(file_key)
    else:
        st.session_state.selected_files.add(file_key)

st.markdown("**Select Models**")
cols = st.columns(min(len(csv_files), 6))
for i, file_path in enumerate(csv_files):
    col_idx = i % len(cols)
    file_key = str(file_path)
    model_name = model_names_map[file_key]
    is_selected = file_key in st.session_state.selected_files

    with cols[col_idx]:
        if is_selected:
            st.button(
                f"✓ {model_name}",
                key=f"btn_{i}",
                use_container_width=True,
                type="primary",
                on_click=toggle_file,
                args=(file_key,)
            )
        else:
            st.button(
                f"{model_name}",
                key=f"btn_{i}",
                use_container_width=True,
                type="secondary",
                on_click=toggle_file,
                args=(file_key,)
            )

selected_files = [Path(f) for f in st.session_state.selected_files]

if not selected_files:
    st.info("Select at least one model to view results.")
    st.stop()

# Load data
@st.cache_data
def load_single_file(results_path):
    results_df = pd.read_csv(results_path)
    return results_df

@st.cache_data
def load_submissions():
    submissions_path = Path("data/submissions.csv")
    if submissions_path.exists():
        return pd.read_csv(
            submissions_path,
            usecols=['id', 'author', 'subreddit', 'title']
        )
    return None

# Load all selected files
datasets = {}
all_personas = set()
all_ids = set()

for file_path in selected_files:
    df = load_single_file(file_path)
    model_name = df['model'].iloc[0] if 'model' in df.columns else file_path.stem
    datasets[model_name] = df

    # Collect personas
    response_cols = [col for col in df.columns if col.startswith("response_")]
    personas = [col.replace("response_", "") for col in response_cols]
    all_personas.update(personas)

    # Collect IDs
    all_ids.update(df['id'].tolist())

# Load submissions metadata
submissions_df = load_submissions()

# Sort personas and models
all_personas = sorted(all_personas)
model_names = sorted(datasets.keys())

# Check if we have metadata
has_metadata = submissions_df is not None

# Get unique subreddits from submissions
subreddits = []
if has_metadata:
    # Filter submissions to only those in our datasets
    relevant_submissions = submissions_df[submissions_df['id'].isin(all_ids)]
    subreddits = sorted(relevant_submissions['subreddit'].dropna().unique().tolist())

# Stats row - all on one line
col1, col2, col3, col4 = st.columns([1, 1.5, 2, 2])
with col1:
    st.markdown(f"**Samples** :gray-background[{len(all_ids)}]")
with col2:
    models_str = " ".join([f":orange-background[{m}]" for m in model_names])
    st.markdown(f"**Models** {models_str}")
with col3:
    personas_str = " ".join([f":green-background[{p}]" for p in all_personas])
    st.markdown(f"**Personas** {personas_str}")
with col4:
    if subreddits:
        subs_str = " ".join([f":blue-background[{s}]" for s in subreddits[:8]])
        extra = f" +{len(subreddits) - 8}" if len(subreddits) > 8 else ""
        st.markdown(f"**Subreddits** {subs_str}{extra}")
    else:
        st.markdown("**Subreddits** —")

st.divider()

# Search/filter
search = st.text_input("Search...", placeholder="Filter by title, subreddit, or content...")

# Build combined dataframe for display (using first dataset as base, merged with metadata)
base_df = list(datasets.values())[0][['id', 'input']].copy()
if has_metadata:
    base_df = base_df.merge(submissions_df, on='id', how='left')

# Filter dataframe
if search:
    mask = base_df["input"].str.contains(search, case=False, na=False)
    if has_metadata:
        mask |= base_df["title"].str.contains(search, case=False, na=False)
        mask |= base_df["subreddit"].str.contains(search, case=False, na=False)
        mask |= base_df["author"].str.contains(search, case=False, na=False)
    filtered_df = base_df[mask].copy()
else:
    filtered_df = base_df.copy()

# Pagination
ITEMS_PER_PAGE = 25
total_items = len(filtered_df)
total_pages = max(1, (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)

if "page" not in st.session_state:
    st.session_state.page = 1
st.session_state.page = min(st.session_state.page, total_pages)

def _sync_page(source_key):
    st.session_state.page = st.session_state[source_key]

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.number_input(
        f"Page (1-{total_pages})",
        min_value=1,
        max_value=total_pages,
        value=st.session_state.page,
        key="page_top",
        on_change=_sync_page,
        args=("page_top",),
    )

page = st.session_state.page
start_idx = (page - 1) * ITEMS_PER_PAGE
end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)

st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_items} samples (Page {page}/{total_pages})")

# Get page slice
page_df = filtered_df.iloc[start_idx:end_idx]

# Display samples
for row_num, (idx, row) in enumerate(page_df.iterrows(), start=start_idx + 1):
    sample_id = row["id"]
    input_text = str(row["input"])

    if has_metadata:
        subreddit = str(row.get("subreddit", ""))
        author = str(row.get("author", ""))
        title = str(row.get("title", ""))
        # Truncate title for preview
        title_preview = title[:80] + "..." if len(title) > 80 else title
        expander_label = f":gray-background[#{row_num}] :blue-background[r/{subreddit}] :green-background[u/{author}] {title_preview}"
    else:
        preview = input_text[:100] + "..." if len(input_text) > 100 else input_text
        preview = preview.replace("\n", " ")
        expander_label = f"**#{row_num}** | {preview}"

    with st.expander(expander_label, expanded=False):
        # Show submission
        st.markdown("##### Submission")
        if has_metadata:
            st.markdown(f"**{title}**")
        st.markdown(f'<div class="submission-content">{input_text}</div>', unsafe_allow_html=True)

        # Show responses in tabs (one tab per persona)
        st.markdown("##### Responses")
        tabs = st.tabs([name.capitalize() for name in all_personas])

        for tab, persona in zip(tabs, all_personas):
            with tab:
                # Show model responses side by side
                cols = st.columns(len(model_names))

                for col, model_name in zip(cols, model_names):
                    with col:
                        st.markdown(f'<div class="model-header">{model_name}</div>', unsafe_allow_html=True)

                        model_df = datasets[model_name]
                        response_col = f"response_{persona}"

                        # Check if this model has this persona
                        if response_col not in model_df.columns:
                            st.markdown('<div class="not-available">Persona not available for this model</div>', unsafe_allow_html=True)
                            continue

                        # Check if this ID exists in this model's dataset
                        model_row = model_df[model_df['id'] == sample_id]
                        if model_row.empty:
                            st.markdown('<div class="not-available">Sample not available for this model</div>', unsafe_allow_html=True)
                            continue

                        response = str(model_row[response_col].iloc[0])
                        st.markdown(f'<div class="model-response">{response}</div>', unsafe_allow_html=True)

# Bottom page selector
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.number_input(
        f"Page (1-{total_pages})",
        min_value=1,
        max_value=total_pages,
        value=st.session_state.page,
        key="page_bottom",
        on_change=_sync_page,
        args=("page_bottom",),
    )
