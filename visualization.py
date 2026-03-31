import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def show_dataset_preview(df):
    """Display a beginner-friendly dataset preview."""
    st.subheader("Dataset Preview")
    st.dataframe(df)


def plot_histogram(df, column, bins=30):
    """Return a histogram plot for one column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(
        data=df,
        x=column,
        bins=bins,
        kde=True,
        ax=ax,
        color="#3b82f6",
        edgecolor="white",
        alpha=0.85,
    )
    col_fmt = column.replace('_', ' ').title()
    ax.set_title(f"Distribution of {col_fmt}", fontsize=16, fontweight="600", color="#1e293b", pad=15)
    ax.set_xlabel(col_fmt, fontsize=13, color="#334155", labelpad=10)
    ax.set_ylabel("Frequency", fontsize=13, color="#334155", labelpad=10)
    ax.tick_params(colors="#475569", labelsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine(left=True)
    fig.tight_layout()
    return fig


def plot_bar_chart(df, x_col, y_col):
    """Return a bar chart using x and y columns."""
    missing = [col for col in [x_col, y_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df, x=x_col, y=y_col, ax=ax, palette="mako", edgecolor="white", alpha=0.9)
    x_fmt = x_col.replace('_', ' ').title()
    y_fmt = y_col.replace('_', ' ').title()
    ax.set_title(f"Comparison of {y_fmt} across {x_fmt}", fontsize=16, fontweight="600", color="#1e293b", pad=15)
    ax.set_xlabel(x_fmt, fontsize=13, color="#334155", labelpad=10)
    ax.set_ylabel(y_fmt, fontsize=13, color="#334155", labelpad=10)
    ax.tick_params(colors="#475569", labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine(left=True)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Return a correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least two numeric columns for correlation heatmap")

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 11, "weight": "bold"}
    )
    nice_labels = [c.replace('_', ' ').title() for c in corr.columns]
    ax.set_xticklabels(nice_labels, rotation=45, ha="right", fontsize=11, color="#334155")
    ax.set_yticklabels(nice_labels, rotation=0, fontsize=11, color="#334155")
    
    ax.set_title("Correlation Heatmap", fontsize=16, fontweight="600", color="#1e293b", pad=20)
    fig.tight_layout()
    return fig


def plot_line_chart(df, x_col, y_col):
    """Return a line chart using x and y columns."""
    missing = [col for col in [x_col, y_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=df, x=x_col, y=y_col, marker="o", ax=ax, color="#10b981", linewidth=3, markersize=8)
    x_fmt = x_col.replace('_', ' ').title()
    y_fmt = y_col.replace('_', ' ').title()
    ax.set_title(f"Trend of {y_fmt} over {x_fmt}", fontsize=16, fontweight="600", color="#1e293b", pad=15)
    ax.set_xlabel(x_fmt, fontsize=13, color="#334155", labelpad=10)
    ax.set_ylabel(y_fmt, fontsize=13, color="#334155", labelpad=10)
    ax.tick_params(colors="#475569", labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    sns.despine(left=True)
    fig.tight_layout()
    return fig
