import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def show_dataset_preview(df):
    """Display a beginner-friendly dataset preview."""
    st.subheader("Dataset Preview")
    st.dataframe(df)


def plot_histogram(df, column, bins=30):
    """Return a histogram plot for one column using pure matplotlib."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    fig, ax = plt.subplots(figsize=(12, 7))
    data = df[column].dropna()
    
    # Gracefully intercept small KDE dataset boundaries natively
    plot_bins = 1 if len(data) < 2 else min(bins, len(data.unique()))
    
    ax.hist(data, bins=plot_bins, color="#3b82f6", edgecolor="white", alpha=0.85)
    
    col_fmt = str(column).replace('_', ' ').title()
    ax.set_title(f"Distribution of {col_fmt}", fontsize=16, fontweight="600", color="#1e293b", pad=15)
    ax.set_xlabel(col_fmt, fontsize=13, color="#334155", labelpad=10)
    ax.set_ylabel("Frequency", fontsize=13, color="#334155", labelpad=10)
    ax.tick_params(colors="#475569", labelsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    fig.tight_layout()
    return fig


def plot_bar_chart(df, x_col, y_col):
    """Return a bar chart using x and y columns natively."""
    missing = [col for col in [x_col, y_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Safely group heavy distributions for Bar Charts natively
    chart_df = df.groupby(x_col)[y_col].mean().reset_index() if len(df) > 20 else df
    
    x_data = chart_df[x_col].astype(str)
    y_data = chart_df[y_col]
    
    ax.bar(x_data, y_data, color="#3b82f6", edgecolor="white", alpha=0.9)
    
    x_fmt = str(x_col).replace('_', ' ').title()
    y_fmt = str(y_col).replace('_', ' ').title()
    ax.set_title(f"Comparison of {y_fmt} across {x_fmt}", fontsize=16, fontweight="600", color="#1e293b", pad=15)
    ax.set_xlabel(x_fmt, fontsize=13, color="#334155", labelpad=10)
    ax.set_ylabel(y_fmt, fontsize=13, color="#334155", labelpad=10)
    ax.tick_params(colors="#475569", labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    fig.tight_layout()
    return fig


def plot_scatter_chart(df, x_col, y_col):
    """Return a scatter plot mathematically tracing 2 continuous boundaries."""
    missing = [col for col in [x_col, y_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
        
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(df[x_col], df[y_col], color="#8b5cf6", alpha=0.7, edgecolor="white", s=80)
    
    x_fmt = str(x_col).replace('_', ' ').title()
    y_fmt = str(y_col).replace('_', ' ').title()
    ax.set_title(f"Scatter Plot of {x_fmt} vs {y_fmt}", fontsize=16, fontweight="600", color="#1e293b", pad=15)
    ax.set_xlabel(x_fmt, fontsize=13, color="#334155", labelpad=10)
    ax.set_ylabel(y_fmt, fontsize=13, color="#334155", labelpad=10)
    ax.tick_params(colors="#475569", labelsize=11)
    ax.grid(linestyle='--', alpha=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Natively renders correlation matrices manually over imshow arrays."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least two numeric columns for correlation heatmap")

    corr = numeric_df.corr().values
    cols = numeric_df.columns
    
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    
    # Native Array Annotations
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr[i, j]
            if pd.notna(val):
                text_color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=11)
                
    # Colorbar Extraction
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11)
    
    nice_labels = [c.replace('_', ' ').title() for c in cols]
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(nice_labels, rotation=45, ha="right", fontsize=11, color="#334155")
    ax.set_yticklabels(nice_labels, rotation=0, fontsize=11, color="#334155")
    
    ax.set_title("Correlation Heatmap", fontsize=16, fontweight="600", color="#1e293b", pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    fig.tight_layout()
    return fig


def plot_line_chart(df, x_col, y_col):
    """Return a line chart entirely driven by matplotlib plot attributes."""
    missing = [col for col in [x_col, y_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df[x_col].astype(str), df[y_col], marker="o", color="#10b981", linewidth=3, markersize=8)
    
    x_fmt = str(x_col).replace('_', ' ').title()
    y_fmt = str(y_col).replace('_', ' ').title()
    ax.set_title(f"Trend of {y_fmt} over {x_fmt}", fontsize=16, fontweight="600", color="#1e293b", pad=15)
    ax.set_xlabel(x_fmt, fontsize=13, color="#334155", labelpad=10)
    ax.set_ylabel(y_fmt, fontsize=13, color="#334155", labelpad=10)
    ax.tick_params(colors="#475569", labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(linestyle='--', alpha=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    fig.tight_layout()
    return fig
