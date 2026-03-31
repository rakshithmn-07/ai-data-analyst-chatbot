import ast
import json
import os
import re
from difflib import get_close_matches

import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _normalize_text(text):
    """Lowercase and remove non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _find_column_from_question(question, df):
    """
    Find best matching dataframe column from question text.
    Matches exact/partial column name in a case-insensitive way.
    Uses fuzzy matching if exact/partial match is not found.
    """
    question_lower = question.lower()
    question_normalized = _normalize_text(question)

    for column in df.columns:
        col_str = str(column)
        col_lower = col_str.lower()
        col_normalized = _normalize_text(col_str)
        if col_lower in question_lower or col_normalized in question_normalized:
            return column

    # Fuzzy matching using difflib to handle slight typos
    candidates = list(map(str, df.columns))
    words = [w for w in question_lower.split() if w.isalnum()]
    
    for word in words:
        if len(word) > 2:
            matches = get_close_matches(word, candidates, n=1, cutoff=0.7)
            if matches:
                return matches[0]

    matches = get_close_matches(question_lower, candidates, n=1, cutoff=0.6)
    if matches:
        return matches[0]

    return None


def _suggest_columns(user_text, df, limit=5):
    """Suggest closest matching column names for a user-provided text."""
    candidates = list(map(str, df.columns))
    matches = get_close_matches(str(user_text), candidates, n=limit, cutoff=0.3)
    if matches:
        return matches

    normalized = _normalize_text(user_text)
    if not normalized:
        return []
    partial = [c for c in candidates if normalized in _normalize_text(c)]
    return partial[:limit]


def _is_id_like_column(col_name):
    """Filter out identification and indexing columns."""
    name = str(col_name).strip().lower()
    id_like = {"id", "index", "student_id", "studentid", "user_id", "userid", "row_id", "rowid"}
    if name in id_like:
        return True
    return name.endswith("_id") or name.startswith("id_") or name.startswith("index_")


def _best_numeric_column(df):
    """Automatically pick the best analytical numeric column."""
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if not _is_id_like_column(c)]
    if not numeric_cols:
        return None
        
    targets = ["math", "science", "english", "attendance_percentage", "attendance", "percentage", "score", "mark"]
    cols_lower = {str(c).lower(): c for c in numeric_cols}
    
    # Priority matching
    for t in targets:
        for k, original in cols_lower.items():
            if t == k:
                return original
        for k, original in cols_lower.items():
            if t in k:
                return original
                
    return numeric_cols[0]


def _closest_column_suggestion(user_text, df):
    """Return a single closest column suggestion using difflib."""
    suggestions = _suggest_columns(user_text, df, limit=1)
    return suggestions[0] if suggestions else None


def _extract_useful_query(raw_question):
    """
    Normalize query for rule-based parsing:
    - map synonyms to canonical words
    - remove filler words
    """
    text = str(raw_question).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    synonym_map = {
        "avg": "mean",
        "average": "mean",
        "maximum": "max",
        "highest": "max",
        "minimum": "min",
        "lowest": "min",
        "distribution": "histogram",
        "visualize": "chart",
    }
    filler_words = {
        "marks",
        "mark",
        "of",
        "the",
        "value",
        "values",
        "please",
        "show",
        "me",
        "is",
        "what",
        "tell",
        "can",
        "you",
        "find",
    }

    tokens = [token for token in text.split() if token]
    mapped_tokens = [synonym_map.get(token, token) for token in tokens]
    useful_tokens = [token for token in mapped_tokens if token not in filler_words]
    return " ".join(useful_tokens)


def _detect_operation(normalized_question):
    """Return canonical operation keyword from normalized text."""
    text = normalized_question
    if "mean" in text:
        return "mean"
    if "max" in text:
        return "max"
    if "min" in text:
        return "min"
    if "median" in text:
        return "median"
    if "mode" in text or "most common" in text:
        return "mode"
    if "sum" in text or "total" in text:
        return "sum"
    if "count" in text:
        return "count"
    return None


def _is_safe_pandas_expression(code):
    """
    Validate that generated code is a safe expression.
    Only allows read-only dataframe/series operations.
    """
    blocked_words = [
        "import",
        "exec",
        "eval",
        "open(",
        "write(",
        "to_csv",
        "to_excel",
        "os.",
        "sys.",
        "__",
    ]
    lowered = code.lower()
    if any(word in lowered for word in blocked_words):
        return False

    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        return False

    allowed_nodes = (
        ast.Expression,
        ast.Load,
        ast.Name,
        ast.Constant,
        ast.Subscript,
        ast.Slice,
        ast.Index,
        ast.Tuple,
        ast.List,
        ast.Dict,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.Attribute,
        ast.Call,
    )

    allowed_root_names = {"df", "pd", "True", "False", "None"}
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return False

        if isinstance(node, ast.Name) and node.id not in allowed_root_names:
            return False

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # Do not allow direct function calls like eval(...)
                return False

    return True


def _result_to_text(result):
    """Convert pandas/python result to readable text."""
    if isinstance(result, pd.DataFrame):
        return result.to_string(max_rows=20)
    if isinstance(result, pd.Series):
        return result.to_string()
    return str(result)


def _explain_simple(operation, column, value):
    """
    Create a simple English explanation for common stats.
    Uses a small heuristic for education-like columns (marks/scores).
    """
    col_text = str(column) if column is not None else "the selected data"
    try:
        numeric_value = float(value)
    except Exception:
        numeric_value = None

    if operation == "mean":
        if numeric_value is None:
            return "This is the typical (average) value in the column."

        # Education-like scale (as requested): >80, 60-80, <60
        if any(key in col_text.lower() for key in ["mark", "marks", "score", "grade"]):
            if numeric_value > 80:
                return "Students are performing very well overall."
            if numeric_value >= 60:
                return "Performance is moderate overall."
            return "Performance needs improvement overall."

        # Generic dynamic interpretation for non-education columns
        if numeric_value > 80:
            return "The average is relatively high, which suggests strong overall values."
        if numeric_value >= 60:
            return "The average is in a mid range, which suggests moderate overall values."
        return "The average is on the lower side, which may indicate room for improvement."

    if operation == "median":
        if numeric_value is not None:
            return (
                f"The middle value is {numeric_value:.2f}, which represents the center of the data "
                "and is less affected by outliers."
            )
        return "This is the middle value when the data is sorted (less affected by outliers)."

    if operation == "mode":
        return "This is the most common value in the column."

    if operation == "max":
        if numeric_value is not None:
            return f"This is the highest observed value ({numeric_value:.2f}) in the column."
        return "This is the highest value found in the column."

    if operation == "min":
        if numeric_value is not None:
            return f"This is the lowest observed value ({numeric_value:.2f}) in the column."
        return "This is the lowest value found in the column."

    if operation == "sum":
        return "This is the total when all values are added together."

    if operation == "count":
        return "This is how many non-missing values are present."

    return "This summarizes the data based on your question."


def _format_result_with_explanation(result_line, explanation):
    return f"{result_line}\nExplanation: {explanation}"


def _find_column_by_name(df, name_text):
    """Match a column by exact/normalized name."""
    target = _normalize_text(name_text)
    for col in df.columns:
        if _normalize_text(col) == target:
            return col
    for col in df.columns:
        if target and target in _normalize_text(col):
            return col
    return None


def _extract_groupby_column(question, df):
    """
    Try to extract a 'group by <column>' column from the question.
    Returns matching df column or None.
    """
    match = re.search(r"group\s*by\s+([a-zA-Z0-9_ ]+)", question, flags=re.IGNORECASE)
    if not match:
        return None

    candidate = match.group(1).strip()
    candidate = re.split(r"(with|for|and|then|where|show|plot|average|mean|max|min|sum|count|median|mode)", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if not candidate:
        return None

    # Try direct match from extracted words
    by_col = _find_column_by_name(df, candidate)
    if by_col is not None:
        return by_col

    # Try best-effort match from entire question
    return _find_column_from_question(candidate, df)


def _extract_filter_query(raw_question, df):
    """
    Look for filter patterns like 'above 80', 'more than 50', 'below 10', 'less than 5'.
    Returns (column, operator, value) or (None, None, None).
    """
    pattern = r"(above|below|greater than|less than|more than|over|under)\s*(\d+(?:\.\d+)?)"
    match = re.search(pattern, raw_question, re.IGNORECASE)
    if not match:
        return None, None, None
        
    keyword = match.group(1).lower()
    value = float(match.group(2))
    
    if keyword in ["above", "greater than", "more than", "over"]:
        operator = ">"
    else:
        operator = "<"
        
    q_stripped = raw_question.replace(match.group(0), "")
    col = _find_column_from_question(q_stripped, df)
    
    if not col:
        col = _best_numeric_column(df)
        
    return col, operator, value


def _extract_range_query(raw_question, df):
    """
    Look for range patterns like 'between 70 and 90'.
    Returns (column, val1, val2) or (None, None, None).
    """
    pattern = r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)"
    match = re.search(pattern, raw_question, re.IGNORECASE)
    if not match:
        return None, None, None
        
    val1 = float(match.group(1))
    val2 = float(match.group(2))
    
    q_stripped = raw_question.replace(match.group(0), "")
    col = _find_column_from_question(q_stripped, df)
    
    if not col:
        col = _best_numeric_column(df)
        
    val_min = min(val1, val2)
    val_max = max(val1, val2)
    
    return col, val_min, val_max


def _extract_exact_query(raw_question, df):
    """
    Look for exact match queries like 'who scored 74 in english', 'got 85', 'marks is 90'.
    Returns (column, value) or (None, None).
    """
    pattern = r"(who scored|which student scored|who got|scored|got|marks is|marks are|is exactly|equals)\s+(\d+(?:\.\d+)?)"
    match = re.search(pattern, raw_question, re.IGNORECASE)
    if not match:
        return None, None
        
    value = float(match.group(2))
    
    q_stripped = raw_question.replace(match.group(0), "")
    col = _find_column_from_question(q_stripped, df)
    
    if not col:
        col = _best_numeric_column(df)
        
    return col, value


def _get_openai_client():
    """Create OpenAI client from OPENAI_API_KEY environment variable."""
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _generate_pandas_code_with_llm(client, question, df):
    """Ask LLM to convert user question into a single pandas expression."""
    system_prompt = (
        "You convert user questions to a SINGLE safe pandas expression using df. "
        "Return ONLY valid JSON in this format: "
        '{"code":"<expression>","note":"<short note>"}. '
        "Rules: use only df and pandas operations, no assignments, no imports, "
        "no file/network/system operations, expression only."
    )

    user_prompt = (
        f"Columns: {list(map(str, df.columns))}\n"
        f"Dtypes: {df.dtypes.astype(str).to_dict()}\n"
        f"Question: {question}\n"
        "Return JSON only."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    parsed = json.loads(content)
    return parsed.get("code", "").strip(), parsed.get("note", "").strip()


def _explain_result_with_llm(client, question, result_text):
    """Generate a beginner-friendly natural language explanation."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Explain data analysis results in simple, beginner-friendly English "
                    "in 2-3 sentences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Result: {result_text}\n"
                    "Explain what this means."
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()


def _answer_question_with_llm(user_question, df):
    """
    LLM flow:
    1) convert question -> pandas expression
    2) validate expression safety
    3) execute expression on dataframe
    4) explain result in natural language
    """
    client = _get_openai_client()
    if client is None:
        return None

    try:
        code, _ = _generate_pandas_code_with_llm(client, user_question, df)
        if not code:
            return None

        if not _is_safe_pandas_expression(code):
            return "Generated code was blocked by safety checks. Please rephrase your question."

        result = eval(code, {"__builtins__": {}}, {"df": df, "pd": pd})  # nosec B307
        result_text = _result_to_text(result)
        explanation = _explain_result_with_llm(client, user_question, result_text)

        return (
            f"Pandas code: {code}\n\n"
            f"Result:\n{result_text}\n\n"
            f"Explanation:\n{explanation}"
        )
    except Exception as error:
        return f"LLM-based analysis failed: {error}"


def answer_question(user_question, df):
    """Answer user question using LLM-to-pandas, with rule-based fallback."""
    if not user_question or not user_question.strip():
        return "Please ask a question about your dataset."

    # Try OpenAI-powered conversion first (if API key is configured).
    llm_response = _answer_question_with_llm(user_question, df)
    if llm_response:
        return llm_response

    try:
        raw_question = user_question.strip().lower()
        question = _extract_useful_query(raw_question)
        operation = _detect_operation(question)

        if any(w in raw_question for w in ["correlation", "heatmap", "relationship"]):
            return _format_result_with_explanation(
                "Correlation Analysis Generated",
                "Here is the correlation analysis for all numeric columns. The chart below visually summarizes the relationships between the variables."
            )

        # Unified Filtering Engine (Range, Comparison, Exact)
        filter_col, filter_op, filter_val = _extract_filter_query(raw_question, df)
        range_col, range_min, range_max = _extract_range_query(raw_question, df)
        exact_col, exact_val = _extract_exact_query(raw_question, df)

        active_filter_type = None
        target_col = None
        
        if range_col is not None and range_min is not None and range_max is not None:
            active_filter_type = "range"
            target_col = range_col
        elif filter_col is not None and filter_op is not None:
            active_filter_type = "comparison"
            target_col = filter_col
        elif exact_col is not None and exact_val is not None:
            active_filter_type = "exact"
            target_col = exact_col

        if active_filter_type:
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                return f"Column '{target_col}' is not numeric, so it cannot be filtered."
                
            if active_filter_type == "range":
                filtered_df = df[(df[target_col] >= range_min) & (df[target_col] <= range_max)]
                desc = f"between {range_min} and {range_max}"
            elif active_filter_type == "comparison":
                if filter_op == ">":
                    filtered_df = df[df[target_col] > filter_val]
                    desc = f"greater than {filter_val}"
                else:
                    filtered_df = df[df[target_col] < filter_val]
                    desc = f"less than {filter_val}"
            else: # exact
                filtered_df = df[df[target_col] == exact_val]
                desc = f"exactly {exact_val}"
                
            name_col = None
            for c in df.columns:
                if str(c).strip().lower() in ["name", "student", "student_name", "user_name"]:
                    name_col = c
                    break
            if not name_col:
                non_numeric = df.select_dtypes(exclude="number").columns
                if len(non_numeric) > 0:
                    name_col = non_numeric[0]
                    
            if filtered_df.empty:
                return _format_result_with_explanation(
                    "No student found." if active_filter_type == "exact" else "No matching records found.",
                    f"No one scored {desc} in {target_col}."
                )

            # Check if query requests names ("who", "which student")
            wants_names = any(w in raw_question for w in ["who", "which student", "whose"])

            # Format 1: Exact string sentence (used for 'exact' or explicitly requesting 'who')
            if active_filter_type == "exact" or wants_names:
                if name_col:
                    names = filtered_df[name_col].astype(str).tolist()
                    name_list = ", ".join(names)
                    result_text = f"**Students who scored {desc} in {target_col}:** {name_list}"
                else:
                    result_text = f"**{len(filtered_df)} records found scoring {desc} in {target_col}** (No name column detected)."
                
                return _format_result_with_explanation(
                    result_text,
                    "Filtering lookup successfully executed."
                )
            
            # Format 2: Dataframe Table (Default for Range & Comparison without "who")
            cols_to_show = []
            if name_col and name_col != target_col:
                cols_to_show.append(name_col)
            cols_to_show.append(target_col)
            
            result_header = f"**Showing rows where {target_col} is {desc}**"
            return _format_result_with_explanation(
                result_header,
                f"Found {len(filtered_df)} row(s) matching your condition."
            ), filtered_df[cols_to_show]

        # Basic group-by (rule-based)
        if "group by" in question:
            by_col = _extract_groupby_column(user_question, df)
            if by_col is None:
                return (
                    "I saw 'group by' but couldn't detect the group column. "
                    f"Try: 'average <numeric_column> group by <category_column>'. Columns: {', '.join(map(str, df.columns))}"
                )

            # Avoid finding the 'group by' column as the 'value' column
            q_without_groupby = re.sub(r"group\s*by\s+.*", "", raw_question, flags=re.IGNORECASE)
            value_col = _find_column_from_question(q_without_groupby, df)
            
            if value_col is None or value_col == by_col:
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != by_col]
                if not numeric_cols:
                    return "No numeric columns found to perform the operation."
                value_col = numeric_cols[0]

            if not pd.api.types.is_numeric_dtype(df[value_col]) and operation not in ["count"]:
                return f"Column '{value_col}' is not numeric, so it cannot be used for '{operation}'."

            if operation in ["mean", "max", "min", "sum", "count"]:
                if operation == "mean":
                    grouped = df.groupby(by_col)[value_col].mean().sort_values(ascending=False).head(10)
                    op_title = f"Average {value_col}"
                elif operation == "max":
                    grouped = df.groupby(by_col)[value_col].max().sort_values(ascending=False).head(10)
                    op_title = f"Maximum {value_col}"
                elif operation == "min":
                    grouped = df.groupby(by_col)[value_col].min().sort_values(ascending=True).head(10)
                    op_title = f"Minimum {value_col}"
                elif operation == "sum":
                    grouped = df.groupby(by_col)[value_col].sum().sort_values(ascending=False).head(10)
                    op_title = f"Sum of {value_col}"
                elif operation == "count":
                    grouped = df.groupby(by_col)[value_col].count().sort_values(ascending=False).head(10)
                    op_title = f"Count of {value_col}"

                result_line = f"**{op_title} grouped by {by_col}** (top 10):\n{grouped.to_string()}"
                explanation = f"This compares the {operation} values of **{value_col}** across different **{by_col}**."
                return _format_result_with_explanation(result_line, explanation)

            return f"For group-by questions, the operation '{operation}' is not fully supported yet. Try 'average', 'max', 'min', 'sum', or 'count'."

        column = _find_column_from_question(raw_question, df) or _find_column_from_question(question, df)

        # Dataset-level quick commands
        if "shape" in question or "size" in question:
            rows, cols = df.shape
            return f"Dataset shape: {rows} rows and {cols} columns."
        if "columns" in question:
            return f"Columns: {', '.join(map(str, df.columns))}"
        if "null" in question or "missing" in question:
            null_counts = df.isnull().sum().to_dict()
            return f"Missing values per column: {null_counts}"
        if "describe" in question or "summary" in question:
            return str(df.describe(include="all"))

        vis_words = ["histogram", "plot", "chart", "graph", "distribution", "analysis", "comparison", "compare", "visualize"]
        wants_vis = any(w in raw_question for w in vis_words)

        # Column-based rules
        if column is None:
            auto_col = _best_numeric_column(df)
            
            if wants_vis:
                if not auto_col:
                    return "I couldn't find any valid numeric columns to generate a visualization."
                col_name = str(auto_col).replace('_', ' ').title()
                return _format_result_with_explanation(
                    "Visualization Triggered",
                    f"No column specified. Showing distribution of {col_name}."
                )
            
            # Intelligent fallback if operation is requested but no column mentioned
            if auto_col is not None and operation in ["mean", "max", "min", "median", "mode", "sum", "count"]:
                column = auto_col
            else:
                suggestion = _closest_column_suggestion(user_question, df)
                if suggestion:
                    return f"I could not detect a column name in your question. Did you mean: **{suggestion}**?"
                return (
                    "I could not detect a column name in your question. "
                    f"Try using one of these columns: {', '.join(map(str, df.columns))}"
                )

        series = df[column]

        if operation == "mean":
            if not pd.api.types.is_numeric_dtype(series):
                return f"Column '{column}' is not numeric, so mean cannot be calculated."
            value = series.mean()
            return _format_result_with_explanation(
                f"Average of {column}: {value}",
                _explain_simple("mean", column, value),
            )

        if operation == "max":
            value = series.max()
            return _format_result_with_explanation(
                f"Maximum of {column}: {value}",
                _explain_simple("max", column, value),
            )

        if operation == "min":
            value = series.min()
            return _format_result_with_explanation(
                f"Minimum of {column}: {value}",
                _explain_simple("min", column, value),
            )

        if operation == "median":
            if not pd.api.types.is_numeric_dtype(series):
                return f"Column '{column}' is not numeric, so median cannot be calculated."
            value = series.median()
            return _format_result_with_explanation(
                f"Median of {column}: {value}",
                _explain_simple("median", column, value),
            )

        if operation == "mode":
            mode_values = series.mode().tolist()
            mode_for_explain = mode_values[0] if mode_values else None
            return _format_result_with_explanation(
                f"Mode of {column}: {mode_values}",
                _explain_simple("mode", column, mode_for_explain),
            )

        if operation == "sum":
            if not pd.api.types.is_numeric_dtype(series):
                return f"Column '{column}' is not numeric, so sum cannot be calculated."
            value = series.sum()
            return _format_result_with_explanation(
                f"Sum of {column}: {value}",
                _explain_simple("sum", column, value),
            )

        if operation == "count":
            value = series.count()
            return _format_result_with_explanation(
                f"Count of non-null values in {column}: {value}",
                _explain_simple("count", column, value),
            )

        if wants_vis:
            return _format_result_with_explanation(
                "Visualization Generated",
                f"I have prepared the visual data for **{column}**. Look at the chart below!"
            )

        # Basic data retrieval (no explicit operation, just list/show)
        retrieval_words = ["list", "show", "give", "display", "get"]
        if any(w in raw_question.split() for w in retrieval_words) and column is not None:
            name_col = None
            for c in df.columns:
                if str(c).strip().lower() in ["name", "student", "student_name", "user_name"]:
                    name_col = c
                    break
            if not name_col:
                non_numeric = df.select_dtypes(exclude="number").columns
                if len(non_numeric) > 0:
                    name_col = non_numeric[0]

            cols_to_show = []
            if name_col and name_col != column:
                cols_to_show.append(name_col)
            cols_to_show.append(column)

            result_header = f"**Data retrieval for {column}**"
            return _format_result_with_explanation(
                result_header,
                f"Displaying top records for {column}."
            ), df[cols_to_show].head(20)

        return (
            "I understand the column, but not the operation yet. "
            "Try words like average, max, min, median, mode, sum, or count."
        )
    except Exception as error:
        return f"Sorry — I couldn't process that query. Details: {error}"
