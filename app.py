import pandas as pd
import streamlit as st
import re

from chatbot import answer_question
from data_analysis import clean_dataframe, get_basic_info, get_preview, load_csv
from visualization import (
    plot_bar_chart,
    plot_correlation_heatmap,
    plot_histogram,
    plot_line_chart,
    show_dataset_preview,
)


def _wants_chart(question):
    text = (question or "").lower()
    chart_words = ["plot", "chart", "graph", "hist", "histogram", "bar", "line", "heatmap", "correlation", "distribution", "relationship", "visualize"]
    return any(word in text for word in chart_words)


def _select_default_column(columns, keyword):
    """Pick a default column if keyword appears in a column name."""
    keyword = keyword.lower()
    for column in columns:
        if keyword in str(column).lower():
            return column
    return columns[0] if columns else None


def _find_column_mentioned(question, df):
    """Try to find a df column name mentioned in the question."""
    text = (question or "").lower()
    for col in df.columns:
        if str(col).lower() in text:
            return col
    return None


def _is_id_like_column(col_name):
    """Return True if column looks like an ID/index column."""
    name = str(col_name).strip().lower()
    id_like = {
        "id",
        "index",
        "student_id",
        "studentid",
        "user_id",
        "userid",
        "row_id",
        "rowid",
    }
    if name in id_like:
        return True
    # Prevent overmatching words like "solid", "grid", "paid"
    return name.endswith("_id") or name.startswith("id_") or name.startswith("index_")


def _meaningful_numeric_columns(df):
    """
    Prefer numeric columns that look meaningful (marks, score, percent, hours, etc.).
    Falls back to all numeric columns except ID-like.
    """
    numeric_cols = [c for c in df.select_dtypes(include="number").columns.tolist() if not _is_id_like_column(c)]
    if not numeric_cols:
        return []

    keywords = ["mark", "marks", "score", "scores", "percent", "percentage", "attendance", "hour", "hours"]
    preferred = [c for c in numeric_cols if any(k in str(c).lower() for k in keywords)]
    return preferred if preferred else numeric_cols


def _preferred_chart_columns(df):
    """
    Prefer common education/attendance columns if present (case-insensitive).
    Example targets: Math, Science, English, Attendance_Percentage.
    """
    targets = ["math", "science", "english", "attendance_percentage", "attendance", "percentage"]
    cols_lower = {str(c).lower(): c for c in df.columns}
    picked = []
    for t in targets:
        for k, original in cols_lower.items():
            if t == k:
                picked.append(original)
        for k, original in cols_lower.items():
            if t in k and original not in picked:
                picked.append(original)
    # Keep only numeric + non-id-like
    picked_numeric = [c for c in picked if c in df.columns and not _is_id_like_column(c)]
    picked_numeric = [c for c in picked_numeric if c in df.select_dtypes(include="number").columns]
    return picked_numeric


def _best_numeric_column_for_chart(df, question=None):
    """
    Choose a strong numeric column for charts:
    1) column mentioned in question (if numeric, non-id-like)
    2) preferred chart columns (Math/Science/English/Attendance...)
    3) meaningful numeric columns (marks/scores/percent/hours...)
    """
    q = (question or "").lower()

    # Keyword-based selection (requested)
    keyword_preferences = []
    if "marks" in q or "mark" in q:
        keyword_preferences.extend(["math", "science", "english"])
    if "attendance" in q:
        keyword_preferences.append("attendance_percentage")
    if "study" in q:
        keyword_preferences.append("study_hours")

    if keyword_preferences:
        lower_to_col = {str(c).lower(): c for c in df.columns}
        numeric_set = set(df.select_dtypes(include="number").columns)
        for pref in keyword_preferences:
            # exact match first
            for k, original in lower_to_col.items():
                if k == pref and original in numeric_set and not _is_id_like_column(original):
                    return original
            # partial match fallback
            for k, original in lower_to_col.items():
                if pref in k and original in numeric_set and not _is_id_like_column(original):
                    return original

    mentioned = _find_column_mentioned(question or "", df)
    if mentioned is not None:
        if mentioned in df.select_dtypes(include="number").columns and not _is_id_like_column(mentioned):
            return mentioned

    preferred = _preferred_chart_columns(df)
    if preferred:
        return preferred[0]

    meaningful = _meaningful_numeric_columns(df)
    return meaningful[0] if meaningful else None


def _best_x_column_for_comparison(df, question=None):
    """Pick a non-numeric, non-id-like X column for comparisons (bar charts)."""
    mentioned = _find_column_mentioned(question or "", df)
    if mentioned is not None and mentioned in df.columns and not _is_id_like_column(mentioned):
        return mentioned

    # Prefer categorical/object columns with reasonable cardinality
    candidates = []
    for c in df.columns:
        if _is_id_like_column(c):
            continue
        if c in df.select_dtypes(include="number").columns:
            continue
        unique_count = df[c].nunique(dropna=True)
        candidates.append((unique_count, c))
    candidates.sort(key=lambda t: t[0])
    if candidates:
        return candidates[0][1]

    # Fallback: any non-id column
    for c in df.columns:
        if not _is_id_like_column(c):
            return c
    return df.columns[0] if len(df.columns) else None


def _suggest_visualization_columns(df, question=None):
    """
    Suggest the best columns for the user to visualize.
    Designed to avoid ID-like columns and reduce random charting.
    """
    text = (question or "").lower()

    def _pick_first_matching(keyword_list):
        for kw in keyword_list:
            for c in df.columns:
                if _is_id_like_column(c):
                    continue
                if kw == str(c).strip().lower():
                    return c
        for kw in keyword_list:
            for c in df.columns:
                if _is_id_like_column(c):
                    continue
                if kw in str(c).lower():
                    return c
        return None

    suggested = []
    for key in ["math", "science", "english"]:
        col = _pick_first_matching([key])
        if col is not None and col not in suggested:
            suggested.append(col)

    attendance_col = _pick_first_matching(["attendance_percentage", "attendance", "percentage"])
    if attendance_col is not None and attendance_col not in suggested:
        suggested.append(attendance_col)

    if ("study" in text or "hours" in text) and len(suggested) < 4:
        study_col = _pick_first_matching(["study_hours", "study hours", "study", "hours"])
        if study_col is not None and study_col not in suggested:
            suggested.append(study_col)

    if not suggested:
        meaningful = _meaningful_numeric_columns(df)
        suggested = meaningful[:4]

    # Keep only non-id columns
    suggested = [c for c in suggested if c in df.columns and not _is_id_like_column(c)]
    return suggested[:4]


def _is_visualization_request_unclear(question, df):
    """
    Decide if a visualization request is unclear (no column clues).
    If unclear, we show suggestions and do not auto-plot random columns.
    """
    text = (question or "").lower()
    
    # Heatmaps/correlations inherently apply to all numeric columns, so no single column is required.
    if any(w in text for w in ["correlation", "heatmap", "relationship"]):
        return False
        
    mentioned = _find_column_mentioned(question, df)
    has_common_topic = any(k in text for k in ["marks", "mark", "attendance", "study", "math", "science", "english"])
    has_comparison = _is_comparison_query(question)
    return mentioned is None and (not has_common_topic) and (not has_comparison)


def _is_comparison_query(question):
    """Detect if user is asking for grouped/comparison style output."""
    text = (question or "").lower()
    comparison_words = ["group by", "compare", "comparison", " vs ", " by "]
    return any(word in text for word in comparison_words)


def _suggest_columns_from_text(text, df, limit=5):
    """Suggest columns based on partial/normalized match (simple, no external deps)."""
    if not text:
        return []
    norm = "".join(ch.lower() for ch in str(text) if ch.isalnum())
    cols = list(map(str, df.columns))
    direct = [c for c in cols if norm and norm in "".join(ch.lower() for ch in c if ch.isalnum())]
    return direct[:limit]


def _render_auto_result_chart(question, df):
    """
    Auto-generate chart after result:
    - comparison query -> bar chart
    - numeric column query -> histogram
    """
    columns = list(df.columns)
    numeric_cols = _meaningful_numeric_columns(df)
    if not columns or not numeric_cols:
        return

    if _is_comparison_query(question):
        x_col = _best_x_column_for_comparison(df, question=question)
        y_col = _best_numeric_column_for_chart(df, question=question) or _select_default_column(numeric_cols, "value")
        fig = plot_bar_chart(df, x_col, y_col)
        st.pyplot(fig)
        st.caption(f"Auto chart: bar chart (X={x_col}, Y={y_col})")
        return

    # Default for numeric analysis questions: histogram
    hist_col = _best_numeric_column_for_chart(df, question=question) or numeric_cols[0]
    fig = plot_histogram(df, hist_col)
    st.pyplot(fig)
    st.caption(f"Auto chart: histogram of {hist_col}")


def _render_chart_from_question(question, df):
    """Create and display chart if question contains chart keywords."""
    text = question.lower()
    columns = list(df.columns)

    if "histogram" in text or "hist" in text or "distribution" in text:
        numeric_cols = _meaningful_numeric_columns(df)
        if not numeric_cols:
            st.warning("No numeric columns found for histogram.")
            return
        default_col = _best_numeric_column_for_chart(df, question=question) or _select_default_column(numeric_cols, "value")
        fig = plot_histogram(df, default_col)
        st.pyplot(fig)
        st.caption(f"Histogram column: {default_col}")
        return

    if "bar" in text or "comparison" in text or "compare" in text:
        if len(columns) < 2:
            st.warning("Need at least two columns for a bar chart.")
            return
        y_candidates = _meaningful_numeric_columns(df) or df.select_dtypes(include="number").columns.tolist()
        if not y_candidates:
            st.warning("No numeric columns found for bar chart.")
            return
        x_guess = _best_x_column_for_comparison(df, question=question) or columns[0]
        y_guess = _best_numeric_column_for_chart(df, question=question) or y_candidates[0]
        fig = plot_bar_chart(df, x_guess, y_guess)
        st.pyplot(fig)
        st.caption(f"Bar chart: X={x_guess}, Y={y_guess}")
        return

    if "line" in text:
        if len(columns) < 2:
            st.warning("Need at least two columns for a line chart.")
            return
        y_candidates = _meaningful_numeric_columns(df) or df.select_dtypes(include="number").columns.tolist()
        if not y_candidates:
            st.warning("No numeric columns found for line chart.")
            return
        x_guess = _best_x_column_for_comparison(df, question=question) or columns[0]
        y_guess = _best_numeric_column_for_chart(df, question=question) or y_candidates[0]
        fig = plot_line_chart(df, x_guess, y_guess)
        st.pyplot(fig)
        st.caption(f"Line chart: X={x_guess}, Y={y_guess}")
        return

    if "heatmap" in text or "correlation" in text or "relationship" in text:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            st.warning("Need at least two numeric columns for correlation analysis.")
            return

        fig = plot_correlation_heatmap(df)
        st.pyplot(fig)

        # Highlight strong correlations (> 0.7)
        corr = numeric_df.corr()
        strong_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if pd.notna(val) and abs(val) > 0.7:
                    strong_pairs.append((corr.columns[i], corr.columns[j], val))

        if strong_pairs:
            explanation = "**💡 Key Correlation Insights (> |0.7|):**\n\n"
            for col1, col2, val in strong_pairs:
                n1 = col1.replace("_", " ").title()
                n2 = col2.replace("_", " ").title()
                if val > 0:
                    explanation += f"- 📈 **{n1}** strongly positively correlates with **{n2}** (score: {val:.2f}). \n  *Meaning: As {n1} increases, {n2} also tends to increase.*\n"
                else:
                    explanation += f"- 📉 **{n1}** strongly negatively correlates with **{n2}** (score: {val:.2f}). \n  *Meaning: As {n1} increases, {n2} tends to decrease.*\n"
            st.info(explanation)
        else:
            st.info("No exceptionally strong correlations (> 0.7 or < -0.7) were found among the numeric columns.")

        return

    # Fallback default chart for "plot" or "chart"
    numeric_cols = _meaningful_numeric_columns(df)
    if numeric_cols:
        default_col = _best_numeric_column_for_chart(df, question=question) or numeric_cols[0]
        fig = plot_histogram(df, default_col)
        st.pyplot(fig)
        st.caption(f"Auto-selected chart: histogram of {default_col}")
    else:
        st.warning("No numeric columns found for default plotting.")


def _display_automatic_insights(df):
    """Generate automatic insights like top/lowest performers and subjects."""
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if not _is_id_like_column(c)]
    if df.empty or not numeric_cols:
        return

    try:
        # Row-wise averages (Performers)
        row_means = df[numeric_cols].mean(axis=1)

        name_col = None
        for col in df.columns:
            if str(col).strip().lower() in ["name", "student", "student_name", "user_name"]:
                name_col = col
                break
        if not name_col:
            non_numeric = df.select_dtypes(exclude="number").columns
            if len(non_numeric) > 0:
                name_col = non_numeric[0]

        top_idx = row_means.idxmax()
        lowest_idx = row_means.idxmin()

        top_name = df.loc[top_idx, name_col] if name_col else f"Row {top_idx + 1}"
        lowest_name = df.loc[lowest_idx, name_col] if name_col else f"Row {lowest_idx + 1}"
        
        insights = []
        insights.append(f"- 🏆 **Top Performer:** {top_name} (Average: {row_means.max():.2f})")
        insights.append(f"- ⚠️ **Lowest Performer:** {lowest_name} (Average: {row_means.min():.2f})")

        # Column-wise averages (Subjects)
        if len(numeric_cols) >= 2:
            col_means = df[numeric_cols].mean(axis=0)
            
            top_subj = col_means.idxmax()
            lowest_subj = col_means.idxmin()
            
            insights.append(f"- 📈 **Highest Average Column:** {str(top_subj).replace('_', ' ').title()} ({col_means.max():.2f})")
            insights.append(f"- 📉 **Lowest Average Column:** {str(lowest_subj).replace('_', ' ').title()} ({col_means.min():.2f})")
            
        st.info("\n".join(insights))
    except Exception:
        # Silently fail to ensure UI doesn't crash on strange dataset shapes
        pass


def _display_query_suggestions(df):
    """Generate and display query suggestions based on available columns."""
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if not _is_id_like_column(c)]
    if not numeric_cols:
        return
        
    def find_col(keywords, default):
        for kw in keywords:
            for c in numeric_cols:
                if kw in str(c).lower():
                    return c
        return default

    col1 = find_col(["science", "score", "mark"], numeric_cols[0])
    col2 = find_col(["math", "percent", "attendance"], numeric_cols[-1] if len(numeric_cols) > 1 else numeric_cols[0])
    col3_1 = find_col(["study", "hour", "time"], numeric_cols[0])
    col3_2 = find_col(["mark", "score", "grade"], numeric_cols[-1] if len(numeric_cols) > 1 else numeric_cols[0])
    
    c1_fmt = str(col1).replace("_", " ")
    c2_fmt = str(col2).replace("_", " ")
    c3_1_fmt = str(col3_1).replace("_", " ")
    c3_2_fmt = str(col3_2).replace("_", " ")

    suggestions = (
        "**💡 Suggested queries to get started:**\n"
        f"- *Try: average {c1_fmt.lower()}*\n"
        f"- *Try: show distribution of {c2_fmt.lower()}*\n"
    )
    if len(numeric_cols) >= 2 and col3_1 != col3_2:
        suggestions += f"- *Try: correlation between {c3_1_fmt.lower()} and {c3_2_fmt.lower()}*\n"
    elif len(numeric_cols) >= 2:
        suggestions += f"- *Try: show correlation heatmap*\n"

    st.markdown(suggestions)


def _split_result_and_explanation(response_text):
    """Split chatbot response into result and explanation parts."""
    text = str(response_text or "")
    marker = "Explanation:"
    if marker in text:
        result_part, explanation_part = text.split(marker, 1)
        return result_part.strip(), explanation_part.strip()
    return text.strip(), ""


def _format_numbers_for_display(text):
    """Round numeric values in text to 2 decimal places."""
    if not text:
        return text

    def _replace(match):
        raw = match.group(0)
        try:
            number = float(raw)
            return f"{number:.2f}"
        except Exception:
            return raw

    # Round plain numeric values while keeping text structure.
    return re.sub(r"(?<![A-Za-z])[-+]?\d*\.\d+|(?<![A-Za-z])[-+]?\d+", _replace, text)


st.set_page_config(page_title="AI Data Analyst Chatbot", page_icon="🤖")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        letter-spacing: 0.3px;
        color: #0f172a;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.4rem;
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 1rem 1rem 0.8rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 0.45rem 0.55rem;
        margin-bottom: 0.8rem;
    }
    div[data-testid="stForm"], div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) {
        border-radius: 12px;
    }
    div[data-testid="stFileUploader"],
    div[data-testid="stTextInput"],
    div[data-testid="stSelectbox"],
    div[data-testid="stRadio"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.45rem 0.65rem;
        margin-bottom: 0.6rem;
    }
    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #2563eb;
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        font-weight: 600;
        padding: 0.45rem 1rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
    }
    div.stButton > button:hover {
        border-color: #1e40af;
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(30, 64, 175, 0.28);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🤖 AI Data Analyst Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload data, clean it, ask questions, and get instant insights with visualizations.</div>',
    unsafe_allow_html=True,
)
st.divider()

def _card_start():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)


def _card_end():
    st.markdown("</div>", unsafe_allow_html=True)


_card_start()
st.header("📁 Upload Dataset")
left_col, right_col = st.columns([1, 1], gap="large")
with left_col:
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.caption("Accepted format: CSV")

with right_col:
    st.subheader("🧹 Data Cleaning")
    missing_strategy = st.radio(
        "Missing value handling",
        options=["fill", "drop"],
        horizontal=True,
        disabled=uploaded_file is None,
    )
    fill_value = 0
    drop_axis = 0
    if missing_strategy == "fill":
        fill_value = st.text_input("Fill value", value="0", disabled=uploaded_file is None)
    else:
        drop_choice = st.selectbox(
            "Drop missing from",
            ["rows", "columns"],
            disabled=uploaded_file is None,
        )
        drop_axis = 0 if drop_choice == "rows" else 1
    st.caption("Tip: Use `fill` for missing numbers or `drop` for strict cleaning.")
_card_end()

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

if uploaded_file is not None:
    try:
        raw_df = load_csv(uploaded_file)

        df = clean_dataframe(
            raw_df,
            missing_strategy=missing_strategy,
            fill_value=fill_value,
            drop_axis=drop_axis,
        )

        with st.sidebar:
            st.title("📁 Dataset Info")
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
            st.divider()
            st.subheader("Column Names")
            st.write(", ".join(map(str, df.columns)))
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        # --- Data Summary Report ---
        _card_start()
        st.header("📋 Data Summary Report")

        st.subheader("Dataset Overview")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Total Rows", df.shape[0])
        with metric_col2:
            st.metric("Total Columns", df.shape[1])
        with metric_col3:
            st.metric("Missing Values", df.isnull().sum().sum())

        with st.expander("Raw Dictionary Info", expanded=False):
            info = get_basic_info(df)
            st.write("Columns:", info["columns"])
            st.write("Null values:", info["null_values"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Key Statistics")
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if not _is_id_like_column(c)]
        if numeric_cols:
            means = df[numeric_cols].mean()
            cols = st.columns(min(len(numeric_cols), 4))
            for i, col in enumerate(numeric_cols[:4]):  # Display up to 4 metrics cleanly
                with cols[i]:
                    st.metric(col.replace('_', ' ').title(), f"{means[col]:.2f}")
        else:
            st.info("No numeric columns available for statistical averages.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Observations")
        _display_automatic_insights(df)

        _card_end()

        # --- Dataset Preview ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
        _card_start()
        st.header("📊 Dataset Preview")
        preview_df = get_preview(df)
        show_dataset_preview(preview_df)
        _card_end()

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
        _card_start()
        st.header("🤖 Chatbot")
        if "question" not in st.session_state:
            st.session_state["question"] = ""

        _display_query_suggestions(df)

        with st.expander("Sample test queries", expanded=False):
            st.caption("Click a button to try a query quickly.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Average of a column"):
                    st.session_state["question"] = "average marks"
                if st.button("Max value"):
                    st.session_state["question"] = "max salary"
            with col2:
                if st.button("Min value"):
                    st.session_state["question"] = "min salary"
                if st.button("Group by analysis"):
                    st.session_state["question"] = "average salary group by department"

            if st.button("Correlation Analysis"):
                st.session_state["question"] = "show correlation heatmap"

        user_question = st.text_input(
            "Ask a question about your dataset",
            placeholder="Example: average marks, max salary, show histogram",
            key="question",
        )

        ask_clicked = st.button("Ask")
        _card_end()
        if ask_clicked:
            response = answer_question(user_question, df)
            result_text, explanation_text = _split_result_and_explanation(response)
            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()
            _card_start()
            st.header("📈 Results")
            st.markdown("<br>", unsafe_allow_html=True)
            formatted_result = _format_numbers_for_display(result_text)
            if formatted_result:
                st.success(formatted_result)
            else:
                st.warning("No result available for this query.")
            if explanation_text:
                st.info(explanation_text)
            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("Chart")
            try:
                if _wants_chart(user_question):
                    _render_chart_from_question(user_question, df)
                else:
                    _render_auto_result_chart(user_question, df)
            except Exception as chart_error:
                # Try to suggest a likely column if user typed one that doesn't exist
                suggestions = _suggest_columns_from_text(user_question, df)
                if suggestions:
                    st.warning(
                        f"Could not generate chart: {chart_error}\n\n"
                        f"Possible columns you can use: {', '.join(suggestions)}"
                    )
                else:
                    st.warning(f"Could not generate chart: {chart_error}")
            _card_end()
    except Exception as error:
        st.warning(f"Could not process the file. Details: {error}")
else:
    st.info("Please upload a CSV file to begin.")
