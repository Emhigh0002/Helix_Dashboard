import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Generate detailed dataset dimensions and stats."""
    return {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "total_cells": int(df.size),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_percentage": float(df.isna().sum().sum() / df.size * 100) if df.size > 0 else 0.0,
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": len(df.select_dtypes(include=['object', 'category']).columns),
        "datetime_cols": len(df.select_dtypes(include=['datetime', 'datetimetz']).columns),
    }

def get_columns_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Generate metadata for each column (type, missing values, unique count, etc.)."""
    records = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = float(null_count / len(df) * 100) if len(df) > 0 else 0.0
        unique_count = int(df[col].nunique())
        dtype = str(df[col].dtype)
        
        sample_vals = df[col].dropna().head(3).tolist()
        sample_str = ", ".join([str(x) for x in sample_vals])
        
        records.append({
            "Column Name": col,
            "Data Type": dtype,
            "Missing Count": null_count,
            "Missing %": round(null_pct, 2),
            "Unique Values": unique_count,
            "Sample Values": sample_str
        })
    return pd.DataFrame(records)

def calculate_correlations(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """Calculate correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return pd.DataFrame()
    return numeric_df.corr(method=method)

# Plotly Visualizations Engine
def plot_histogram(df: pd.DataFrame, x_col: str, color_col: str = None, bins: int = 30) -> go.Figure:
    """Plot an interactive histogram/density chart."""
    fig = px.histogram(
        df, 
        x=x_col, 
        color=color_col, 
        nbins=bins, 
        marginal="box",
        barmode="overlay",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_box(df: pd.DataFrame, y_col: str, x_col: str = None, color_col: str = None) -> go.Figure:
    """Plot an interactive box plot."""
    fig = px.box(
        df, 
        y=y_col, 
        x=x_col, 
        color=color_col, 
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, size_col: str = None) -> go.Figure:
    """Plot an interactive scatter plot."""
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col, 
        size=size_col, 
        trendline="ols" if (df[x_col].dtype in [np.float64, np.int64] and df[y_col].dtype in [np.float64, np.int64] and df[x_col].isna().sum() == 0 and df[y_col].isna().sum() == 0) else None,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_bar(df: pd.DataFrame, x_col: str, y_col: str = None, color_col: str = None, barmode: str = "group") -> go.Figure:
    """Plot an interactive bar chart."""
    if y_col is None:
        # Value counts bar chart
        value_counts = df[x_col].value_counts().reset_index()
        value_counts.columns = [x_col, 'Count']
        fig = px.bar(
            value_counts, 
            x=x_col, 
            y='Count', 
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
    else:
        fig = px.bar(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_col, 
            barmode=barmode,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_line(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None) -> go.Figure:
    """Plot an interactive line chart."""
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col, 
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_pie(df: pd.DataFrame, names_col: str, values_col: str = None) -> go.Figure:
    """Plot an interactive pie chart."""
    if values_col is None:
        value_counts = df[names_col].value_counts().reset_index()
        value_counts.columns = [names_col, 'Count']
        fig = px.pie(
            value_counts, 
            names=names_col, 
            values='Count', 
            hole=0.4,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
    else:
        fig = px.pie(
            df, 
            names=names_col, 
            values=values_col, 
            hole=0.4,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_violin(df: pd.DataFrame, y_col: str, x_col: str = None, color_col: str = None) -> go.Figure:
    """Plot an interactive violin plot."""
    fig = px.violin(
        df, 
        y=y_col, 
        x=x_col, 
        color=color_col, 
        box=True, 
        points="all", 
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Plot a correlation matrix heatmap."""
    if corr_matrix.empty:
        # Empty heatmap placeholder
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns for correlation", showarrow=False, font=dict(size=16))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig
        
    z = corr_matrix.values
    x = corr_matrix.columns.tolist()
    y = corr_matrix.index.tolist()
    
    # Format hover text
    hover_text = []
    for i in range(len(y)):
        row_text = []
        for j in range(len(x)):
            row_text.append(f"Var 1: {y[i]}<br>Var 2: {x[j]}<br>Correlation: {z[i][j]:.3f}")
        hover_text.append(row_text)
        
    fig = ff.create_annotated_heatmap(
        z, 
        x=x, 
        y=y, 
        annotation_text=np.round(z, 2).astype(str),
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        hoverinfo='text',
        text=hover_text
    )
    fig.update_layout(
        title_text="Correlation Heatmap",
        title_x=0.5,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=40, t=60, b=40)
    )
    # Style annotations font colors
    for annotation in fig.layout.annotations:
        annotation.font.color = 'black' if abs(float(annotation.text)) > 0.4 else 'white'
        annotation.font.size = 10
    return fig

def plot_3d_scatter(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, color_col: str = None, size_col: str = None) -> go.Figure:
    """Plot an interactive 3D scatter plot."""
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        size=size_col,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)")
        )
    )
    return fig

def plot_area(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None) -> go.Figure:
    """Plot an interactive area chart."""
    fig = px.area(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_sunburst(df: pd.DataFrame, path_cols: list[str], values_col: str = None) -> go.Figure:
    """Plot an interactive Sunburst chart."""
    fig = px.sunburst(
        df,
        path=path_cols,
        values=values_col,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_splom(df: pd.DataFrame, dimensions: list[str], color_col: str = None) -> go.Figure:
    """Plot a Scatter Plot Matrix (SPLOM)."""
    fig = px.scatter_matrix(
        df,
        dimensions=dimensions,
        color=color_col,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_traces(diagonal_visible=False)
    return fig

