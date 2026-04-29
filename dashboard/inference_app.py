import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
from markdown_it import MarkdownIt


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = REPO_ROOT / "outputs" / "inference"
PAGE_SIZE_OPTIONS = [10, 25, 50, 100]
MARKDOWN_RENDERER = MarkdownIt("commonmark", {"breaks": True, "html": False})


@dataclass(frozen=True)
class InferenceRun:
    config_name: str
    run_name: str
    split_name: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.config_name} / {self.run_name} / {self.split_name}"


def _relative_path(path: Path | None) -> str:
    if path is None:
        return "Not available"
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def discover_inference_runs() -> list[InferenceRun]:
    runs = []
    for path in sorted(INFERENCE_DIR.rglob("*.jsonl")):
        relative_parts = path.relative_to(INFERENCE_DIR).parts
        if len(relative_parts) < 2:
            continue
        config_name = relative_parts[0]
        run_name = "/".join(relative_parts[1:-1]) or "."
        runs.append(
            InferenceRun(
                config_name=config_name,
                run_name=run_name,
                split_name=path.stem,
                path=path,
            )
        )
    return runs


@st.cache_data(show_spinner=False)
def load_jsonl(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_messages(value: object, *, default_role: str) -> list[dict[str, str]]:
    if isinstance(value, list):
        messages: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            messages.append(
                {
                    "role": str(item.get("role", default_role)),
                    "content": str(item.get("content", "")).strip(),
                }
            )
        if messages:
            return messages
    if value is None:
        return []
    return [{"role": default_role, "content": str(value).strip()}]


def _messages_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(message["content"] for message in messages if message["content"])


def _messages_search_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{message['role']}: {message['content']}" for message in messages if message["content"]
    )


def _prompt_preview(messages: list[dict[str, str]]) -> str:
    text = _messages_text(messages).replace("\n", " ").strip()
    if len(text) > 120:
        return text[:117] + "..."
    return text


def _normalize_inference_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt_messages = _normalize_messages(row.get("prompt"), default_role="user")
    generated_messages = _normalize_messages(row.get("generated"), default_role="assistant")
    metadata = {
        str(key): value
        for key, value in row.items()
        if str(key) not in {"prompt", "generated", "chosen", "rejected"}
    }
    return {
        "prompt_messages": prompt_messages,
        "generated_messages": generated_messages,
        "prompt_text": _messages_text(prompt_messages),
        "generated_text": _messages_text(generated_messages),
        "metadata": metadata,
    }


@st.cache_data(show_spinner=False)
def load_inference_run(path_str: str) -> dict[str, Any]:
    rows = [_normalize_inference_row(row) for row in load_jsonl(path_str)]
    return {
        "rows": rows,
    }


def _matches_search(row: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    lowered = query.casefold()
    searchable = " ".join(
        [
            row["prompt_text"],
            row["generated_text"],
            _metadata_search_text(row["metadata"]),
        ]
    ).casefold()
    return lowered in searchable


def _filter_rows(rows: list[dict[str, Any]], *, search_query: str) -> list[dict[str, Any]]:
    return [row for row in rows if _matches_search(row, search_query)]


def _metadata_search_text(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in metadata.items():
        parts.append(f"{key}: {_format_metadata_value(value)}")
    return "\n".join(parts)


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _render_stat_card(label: str, value: int | str) -> str:
    return (
        '<div class="stat-card">'
        f'<div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(str(value))}</div>'
        "</div>"
    )


def _render_note_card(title: str, body: str) -> str:
    return (
        '<div class="note-card">'
        f'<div class="note-title">{html.escape(title)}</div>'
        f'<div class="note-body">{html.escape(body)}</div>'
        "</div>"
    )


def _render_message_card(title: str, content: str, variant: str) -> str:
    rendered_content = MARKDOWN_RENDERER.render(content)
    return (
        f'<div class="message-card {html.escape(variant)}">'
        f'<div class="message-title">{html.escape(title)}</div>'
        f'<div class="message-body">{rendered_content}</div>'
        "</div>"
    )


def _render_messages_section(messages: list[dict[str, str]], variant: str) -> None:
    for idx, message in enumerate(messages, start=1):
        role = message["role"].capitalize() or "Message"
        label = role if len(messages) == 1 else f"{role} {idx}"
        st.markdown(
            _render_message_card(label, message["content"], variant),
            unsafe_allow_html=True,
        )


st.set_page_config(
    page_title="Inference Viewer",
    page_icon=":bookmark_tabs:",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --paper-bg: #f3efe7;
            --paper-wash-top: #f8f4ed;
            --paper-wash-bottom: #f3efe7;
            --paper-glow: rgba(164, 75, 26, 0.12);
            --panel-primary: #fffaf2;
            --panel-secondary: #f7f0e3;
            --border: #d6c7ae;
            --ink: #221f1a;
            --muted: #6b6254;
            --burnt-orange: #a44b1a;
            --muted-teal: #1e6b6f;
            --user-card: #fff2d9;
            --model-card: #eef7f8;
            --reference-card: #f3eee4;
            --shadow: 0 10px 30px rgba(34, 31, 26, 0.08);
            --body-font: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
            --mono-font: "SFMono-Regular", "Menlo", "Monaco", monospace;
        }
        .stApp {
            background:
                radial-gradient(circle at top right, var(--paper-glow) 0%, rgba(164, 75, 26, 0) 40%),
                linear-gradient(180deg, var(--paper-wash-top) 0%, var(--paper-wash-bottom) 100%);
            color: var(--ink);
            font-family: var(--body-font);
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background:
                radial-gradient(circle at 12% 8%, rgba(164, 75, 26, 0.08) 0%, rgba(164, 75, 26, 0) 22%),
                radial-gradient(circle at 88% 14%, rgba(30, 107, 111, 0.06) 0%, rgba(30, 107, 111, 0) 18%);
            z-index: 0;
        }
        .stApp > header {
            background: transparent;
        }
        .stApp, .stApp p, .stApp label, .stApp div {
            font-family: var(--body-font);
            color: var(--ink);
        }
        .material-symbols-rounded,
        .material-icons {
            font-family: "Material Symbols Rounded", "Material Icons" !important;
        }
        .block-container {
            max-width: 1480px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            position: relative;
            z-index: 1;
        }
        h1, h2, h3 {
            font-family: var(--body-font);
            color: var(--ink);
            letter-spacing: 0.01em;
        }
        h1 {
            font-size: 1.95rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            line-height: 1.02;
        }
        [data-testid="stSidebar"] {
            background: rgba(255, 250, 242, 0.76);
            backdrop-filter: blur(16px);
            border-right: 1px solid rgba(214, 199, 174, 0.8);
        }
        [data-testid="stSidebar"] > div:first-child {
            background: transparent;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--muted);
        }
        [data-testid="stSidebar"] label {
            color: var(--muted);
            font-size: 0.92rem;
        }
        .hero-panel,
        .note-card,
        .stat-card,
        [data-testid="stExpander"] details {
            background: var(--panel-primary);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: var(--shadow);
        }
        .hero-panel {
            padding: 0.9rem 1.15rem 0.85rem 1.15rem;
            margin-top: 0.4rem;
            margin-bottom: 1.35rem;
            background:
                linear-gradient(180deg, rgba(255, 250, 242, 0.96) 0%, rgba(247, 240, 227, 0.96) 100%);
        }
        .eyebrow {
            color: var(--muted-teal);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.8rem;
            margin-bottom: 0.35rem;
            line-height: 1.3;
            padding-top: 0.1rem;
        }
        .hero-copy {
            color: var(--muted);
            max-width: 70rem;
            font-size: 0.98rem;
            line-height: 1.45;
        }
        .note-card {
            padding: 0.8rem 0.95rem;
            min-height: 100%;
        }
        .note-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted-teal);
            margin-bottom: 0.35rem;
        }
        .note-body {
            color: var(--muted);
            font-size: 0.94rem;
            overflow-wrap: anywhere;
        }
        .stat-card {
            padding: 0.75rem 0.95rem;
            min-height: 100%;
            margin-bottom: 0.9rem;
            background: linear-gradient(180deg, rgba(255, 250, 242, 0.98) 0%, rgba(247, 240, 227, 0.9) 100%);
        }
        .stat-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
            margin-bottom: 0.3rem;
        }
        .stat-value {
            color: var(--ink);
            font-size: 1.55rem;
            line-height: 1;
        }
        [data-testid="column"] {
            display: flex;
        }
        [data-testid="column"] > div {
            width: 100%;
        }
        [data-testid="stExpander"] {
            margin-bottom: 1rem;
        }
        [data-testid="stExpander"] details {
            overflow: hidden;
            background: linear-gradient(180deg, rgba(255, 250, 242, 0.98) 0%, rgba(247, 240, 227, 0.96) 100%);
        }
        [data-testid="stExpander"] summary {
            padding: 0.95rem 1.05rem;
            background: transparent;
            color: var(--ink);
        }
        [data-testid="stExpander"] summary:hover {
            color: var(--burnt-orange);
        }
        [data-testid="stExpander"] summary p {
            font-size: 1rem;
            line-height: 1.45;
            white-space: pre-line;
            margin: 0;
        }
        .message-card {
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
            min-height: 100%;
            margin-bottom: 0.8rem;
        }
        .message-card.prompt {
            background: var(--user-card);
        }
        .message-card.generated {
            background: linear-gradient(180deg, var(--model-card) 0%, #f6fbfb 100%);
            border-color: rgba(30, 107, 111, 0.28);
        }
        .message-card.chosen {
            background: var(--reference-card);
        }
        .message-card.rejected {
            background: linear-gradient(180deg, #f5f0e7 0%, #fbf8f2 100%);
        }
        .message-title {
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted-teal);
            margin-bottom: 0.65rem;
        }
        .message-body {
            color: var(--ink);
        }
        .message-body > :first-child {
            margin-top: 0;
        }
        .message-body > :last-child {
            margin-bottom: 0;
        }
        .message-body p,
        .message-body li,
        .message-body blockquote,
        .message-body pre,
        .message-body code {
            font-family: var(--mono-font);
        }
        .message-body p,
        .message-body li,
        .message-body blockquote {
            font-size: 0.91rem;
            line-height: 1.68;
        }
        .message-body p {
            margin: 0 0 0.8rem 0;
        }
        .message-body ul,
        .message-body ol {
            margin: 0.15rem 0 0.9rem 1.35rem;
            padding: 0;
        }
        .message-body li + li {
            margin-top: 0.35rem;
        }
        .message-body code {
            font-size: 0.88rem;
            background: rgba(34, 31, 26, 0.05);
            padding: 0.08rem 0.28rem;
            border-radius: 6px;
        }
        .message-body pre {
            margin: 0 0 0.9rem 0;
            background: rgba(255, 250, 242, 0.72);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.8rem 0.9rem;
            overflow-x: auto;
        }
        .message-body pre code {
            background: transparent;
            padding: 0;
        }
        .message-body blockquote {
            margin: 0 0 0.9rem 0;
            padding-left: 0.9rem;
            border-left: 3px solid rgba(30, 107, 111, 0.35);
            color: var(--muted);
        }
        .meta-chip {
            display: inline-block;
            padding: 0.16rem 0.58rem;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            border-radius: 999px;
            background: rgba(214, 199, 174, 0.42);
            color: var(--muted);
            font-size: 0.82rem;
        }
        .section-gap {
            margin-top: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .page-control {
            margin: 1rem 0 0.35rem 0;
            padding-top: 0.1rem;
        }
        .page-caption {
            margin-bottom: 0.8rem;
            color: var(--muted);
        }
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stMultiSelect"] > div > div,
        [data-testid="stTextInput"] input {
            background: rgba(255, 250, 242, 0.95);
            border: 1px solid var(--border);
            color: var(--ink);
            border-radius: 12px;
        }
        [data-testid="stNumberInput"] [data-baseweb="input"] {
            background: rgba(255, 250, 242, 0.95);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }
        [data-testid="stNumberInput"] input {
            background: transparent;
            border: none;
            box-shadow: none;
            color: var(--ink);
        }
        [data-testid="stNumberInput"] button {
            background: rgba(255, 250, 242, 0.95);
            border: none;
            box-shadow: none;
            color: var(--ink);
        }
        [data-testid="stNumberInput"] {
            max-width: 12rem;
        }
        [data-testid="stCheckbox"] label {
            color: var(--muted);
        }
        a, a:visited {
            color: var(--muted-teal);
        }
        a:hover {
            color: var(--burnt-orange);
        }
        .stInfo, .stWarning {
            background: rgba(255, 250, 242, 0.9);
            border: 1px solid var(--border);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-panel">
        <div class="eyebrow">Inference Ledger</div>
        <h1>Character Inference Viewer</h1>
        <div class="hero-copy">
            Inspect validation-time model generations against the prompt and the chosen/rejected
            reference responses for a character-trained run.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

runs = discover_inference_runs()
if not runs:
    st.warning("No inference JSONL runs found under outputs/inference.")
    st.stop()

run_options = {run.label: run for run in runs}
st.sidebar.markdown("### Inference Index")
st.sidebar.caption("Choose an inference run, search its rows, and inspect attached metadata.")
selected_label = st.sidebar.selectbox("Run", list(run_options))
selected_run = run_options[selected_label]

run_data = load_inference_run(str(selected_run.path))
rows = run_data["rows"]

search_query = st.sidebar.text_input("Search", placeholder="Prompt, generated text, or metadata")
page_size = st.sidebar.selectbox("Rows per page", PAGE_SIZE_OPTIONS, index=1)

filtered_rows = _filter_rows(rows, search_query=search_query)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
metric_cols = st.columns(4)
metric_cols[0].markdown(_render_stat_card("Visible Rows", len(filtered_rows)), unsafe_allow_html=True)
metric_cols[1].markdown(
    _render_stat_card("Run", selected_run.run_name),
    unsafe_allow_html=True,
)
metric_cols[2].markdown(
    _render_stat_card("Split", selected_run.split_name),
    unsafe_allow_html=True,
)
metric_cols[3].markdown(
    _render_stat_card("Rows Total", len(rows)),
    unsafe_allow_html=True,
)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
note_cols = st.columns(3)
with note_cols[0]:
    st.markdown(
        _render_note_card("Inference file", _relative_path(selected_run.path)),
        unsafe_allow_html=True,
    )
with note_cols[1]:
    st.markdown(
        _render_note_card("Run", selected_run.label),
        unsafe_allow_html=True,
    )
with note_cols[2]:
    st.markdown(
        _render_note_card("Search", search_query or "No search filter"),
        unsafe_allow_html=True,
    )

if not filtered_rows:
    st.info("No rows matched the current filters.")
    st.stop()

total_pages = max(1, (len(filtered_rows) + page_size - 1) // page_size)
st.markdown('<div class="page-control">', unsafe_allow_html=True)
page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
st.markdown("</div>", unsafe_allow_html=True)
start = (page - 1) * page_size
end = min(start + page_size, len(filtered_rows))

st.markdown(
    f'<div class="page-caption">Showing {start + 1}-{end} of {len(filtered_rows)} rows</div>',
    unsafe_allow_html=True,
)

for row_index, row in enumerate(filtered_rows[start:end], start=start + 1):
    label = (
        f"#{row_index}\n"
        f"Prompt: {_prompt_preview(row['prompt_messages'])}"
    )

    with st.expander(label):
        st.subheader("Prompt")
        _render_messages_section(row["prompt_messages"], "prompt")

        st.subheader("Generated")
        _render_messages_section(row["generated_messages"], "generated")

        st.subheader("Metadata")
        if not row["metadata"]:
            st.caption("No metadata attached.")
        else:
            for key, value in row["metadata"].items():
                with st.expander(key):
                    st.code(_format_metadata_value(value), language="text")
