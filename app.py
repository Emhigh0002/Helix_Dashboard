import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import custom modules
import data_cleaning as dc
import eda
import ml_module as ml
import utils

# Set Page Config
st.set_page_config(
    page_title="Emhigh_Analytic",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Storage & Projects Cache
utils.init_cache()

# Load Custom Theme/Styles
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium Glassmorphic Card Styling */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* Vibrant Gradient Titles */
    .gradient-title {
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Stats Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 5px;
    }
    
    .metric-label {
        font-size: 13px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Main App Navigation & Logic
def main():
    inject_custom_css()
    
    # Initialize Session State
    if 'active_project_id' not in st.session_state:
        st.session_state.active_project_id = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = None
    if 'notes' not in st.session_state:
        st.session_state.notes = ""
        
    projects = utils.load_projects_metadata()
    
    # Sidebar Logo and Navigation Title
    st.sidebar.markdown("<h2 class='gradient-title' style='text-align: center;'>🧬 Emhigh_Analytic</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # --- Project Management Section ---
    st.sidebar.subheader("📂 Workspace Projects")
    project_options = ["None (Select or Create)"] + list(projects.keys())
    
    # Select active project
    selected_proj = st.sidebar.selectbox("Active Project", options=project_options)
    
    if selected_proj != "None (Select or Create)":
        if st.session_state.active_project_id != selected_proj:
            st.session_state.active_project_id = selected_proj
            # Load project DataFrame
            file_path = projects[selected_proj]["file_path"]
            try:
                st.session_state.df = utils.load_dataset(file_path)
                st.session_state.ml_results = None
            except Exception as e:
                st.sidebar.error(f"Error loading dataset: {e}")
                st.session_state.df = None
    else:
        st.session_state.active_project_id = None
        st.session_state.df = None
        
    # Create new project dialog
    with st.sidebar.expander("➕ Create New Project"):
        new_proj_name = st.text_input("Project Name")
        uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "json", "parquet"])
        
        if st.button("Initialize Project"):
            if new_proj_name and uploaded_file:
                if new_proj_name in projects:
                    st.error("Project with this name already exists.")
                else:
                    # Save uploaded file temporarily to local directory
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    temp_dir = os.path.join(base_dir, "temp_uploads")
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                    
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Load DF & Save as Parquet cache
                        df_temp = utils.load_dataset(temp_path)
                        cached_path = utils.save_dataframe_to_cache(df_temp, uploaded_file.name)
                        
                        # Save to metadata
                        projects[new_proj_name] = {
                            "created_at": datetime.now().isoformat(),
                            "original_name": uploaded_file.name,
                            "file_path": cached_path,
                            "cleaning_log": []
                        }
                        utils.save_projects_metadata(projects)
                        st.session_state.active_project_id = new_proj_name
                        st.session_state.df = df_temp
                        st.session_state.ml_results = None
                        st.success(f"Project '{new_proj_name}' created!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
            else:
                st.warning("Please specify project name and dataset file.")
                
    st.sidebar.markdown("---")
    
    # --- Navigation Tabs ---
    st.sidebar.subheader("🗺️ Module Navigation")
    menu = ["📊 Overview Dashboard", "🧹 Data Cleaning", "📈 Interactive Visuals", "🧮 Statistical Models", "🤖 AutoML Hub", "📝 Summaries & Reports", "💬 AI Assistant"]
    choice = st.sidebar.radio("Go to", menu, disabled=(st.session_state.df is None))
    
    if st.session_state.df is None:
        # Prompt user to initialize/select project
        st.markdown("<div class='glass-card' style='text-align: center; margin-top: 50px;'>", unsafe_allow_html=True)
        st.markdown("<h1 class='gradient-title'>🧬 Welcome to Emhigh_Analytic</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:18px; color:#94A3B8;'>Upload and explore datasets, clean outliers, compute statistical models, run AutoML predictors, and compile downloadable reports in a modern, dark-themed workspace.</p>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#F1F5F9;'>To get started, please select or create a Project in the sidebar.</h4>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display some screenshots/visual templates in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='glass-card'><h4>🧹 Data Cleaning Studio</h4><p style='font-size:14px; color:#94A3B8;'>Directly drop/impute missing cells, crop statistical outliers, scale columns, and convert structural types with complete undo tracking.</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='glass-card'><h4>📊 Interactive Chart Builders</h4><p style='font-size:14px; color:#94A3B8;'>Generate correlation matrices, box plots, histograms, scatters, and violins, customized dynamically via Plotly.</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='glass-card'><h4>🤖 Auto-ML Engines</h4><p style='font-size:14px; color:#94A3B8;'>Train regression or classification algorithms in one click, compare test performance, and view feature importances.</p></div>", unsafe_allow_html=True)
        return
        
    df = st.session_state.df
    active_proj = st.session_state.active_project_id
    
    # --- Menu 1: Overview Dashboard ---
    if choice == "📊 Overview Dashboard":
        st.markdown(f"<h1>📊 Dataset Overview: <span class='gradient-title'>{active_proj}</span></h1>", unsafe_allow_html=True)
        
        # Get metrics
        summary = eda.get_dataset_summary(df)
        
        # Stats Metric Cards
        st.markdown(f"""
        <div class='metric-container'>
            <div class='metric-card'>
                <div class='metric-label'>Total Rows</div>
                <div class='metric-value'>{summary['num_rows']}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Total Columns</div>
                <div class='metric-value'>{summary['num_cols']}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Numeric Columns</div>
                <div class='metric-value'>{summary['numeric_cols']}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Categorical Columns</div>
                <div class='metric-value'>{summary['categorical_cols']}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Missing Cells</div>
                <div class='metric-value'>{summary['missing_cells']} ({summary['missing_percentage']:.1f}%)</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Duplicate Rows</div>
                <div class='metric-value'>{summary['duplicate_rows']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("<div class='glass-card'><h3>🔍 Raw Preview (First 10 Rows)</h3></div>", unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            
        with col2:
            st.markdown("<div class='glass-card'><h3>⚙️ Column Metadata Specifications</h3></div>", unsafe_allow_html=True)
            meta_df = eda.get_columns_metadata(df)
            st.dataframe(meta_df, use_container_width=True, hide_index=True)
            
        # Display Correlation Heatmap on Overview
        st.markdown("<div class='glass-card'><h3>🔥 Numerical Correlations Matrix</h3></div>", unsafe_allow_html=True)
        corr_matrix = eda.calculate_correlations(df)
        if not corr_matrix.empty:
            fig = eda.plot_heatmap(corr_matrix)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found for correlation matrix.")
            
    # --- Menu 2: Data Cleaning ---
    elif choice == "🧹 Data Cleaning":
        st.markdown(f"<h1>🧹 Data Cleaning Studio: <span class='gradient-title'>{active_proj}</span></h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8;'>Impute missing values, drop duplicates, cap outliers, convert features, or scale metrics dynamically.</p>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["💧 Manage Missing/Duplicates", "📈 Outlier Trimmer", "📏 Scaling & Conversion", "📜 Cleaning Log"])
        
        # Keep track of local operations to save
        updated_df = df.copy()
        log_entry = None
        
        with tab1:
            st.subheader("Remove Duplicates")
            dup_count = df.duplicated().sum()
            st.write(f"Current duplicate row count: **{dup_count}**")
            if dup_count > 0:
                if st.button("Drop Duplicate Rows"):
                    updated_df, removed = dc.clean_duplicates(df)
                    log_entry = f"Dropped {removed} duplicate rows."
                    st.success(log_entry)
                    
            st.write("---")
            st.subheader("Impute Missing Values")
            null_cols = df.columns[df.isna().any()].tolist()
            if not null_cols:
                st.info("No missing values detected in the dataset!")
            else:
                col_to_impute = st.multiselect("Columns to Clean", options=null_cols)
                impute_strategy = st.selectbox("Imputation Strategy", options=["drop", "mean", "median", "mode", "constant"])
                fill_val = None
                if impute_strategy == "constant":
                    fill_val = st.text_input("Constant value to fill")
                    
                if st.button("Apply Imputation"):
                    if col_to_impute:
                        updated_df, summary = dc.clean_missing_values(df, impute_strategy, col_to_impute, fill_val)
                        log_entry = f"Applied missing values impute ({impute_strategy}) on columns {col_to_impute}."
                        st.success("Successfully updated DataFrame!")
                        st.json(summary)
                    else:
                        st.warning("Please select at least one column.")
                        
        with tab2:
            st.subheader("IQR Outlier Filtering")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.info("No numeric columns found for outlier operations.")
            else:
                outlier_col = st.selectbox("Column to scan for outliers", options=numeric_cols)
                iqr_k = st.slider("IQR threshold multiplier (k)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
                outlier_method = st.selectbox("Handling Method", options=["cap", "drop"])
                
                # Scan for preview
                try:
                    outliers, lower_b, upper_b = dc.detect_outliers_iqr(df, outlier_col, iqr_k)
                    st.write(f"Scanned lower bound: `{lower_b:.2f}`, upper bound: `{upper_b:.2f}`")
                    st.write(f"Detected **{outliers.sum()}** outlier rows ({outliers.sum() / len(df) * 100:.2f}%)")
                    
                    if outliers.sum() > 0:
                        if st.button("Handle Outliers"):
                            updated_df, handled = dc.handle_outliers(df, outlier_col, outlier_method, iqr_k)
                            log_entry = f"Handled {handled} outliers in '{outlier_col}' via '{outlier_method}'."
                            st.success(log_entry)
                except Exception as e:
                    st.error(f"Scan error: {e}")
                    
        with tab3:
            st.subheader("Feature Conversion")
            conv_col = st.selectbox("Select Column to Convert", options=df.columns)
            target_t = st.selectbox("Target Data Type", options=["numeric", "categorical", "datetime"])
            if st.button("Convert Type"):
                updated_df = dc.convert_column_type(df, conv_col, target_t)
                log_entry = f"Converted column '{conv_col}' type to '{target_t}'."
                st.success(log_entry)
                
            st.write("---")
            st.subheader("Feature Scaling")
            scale_cols = st.multiselect("Select columns to normalize", options=numeric_cols)
            scale_method = st.selectbox("Scaling Method", options=["standard", "minmax"])
            if st.button("Apply Scaling"):
                if scale_cols:
                    updated_df = dc.scale_features(df, scale_cols, scale_method)
                    log_entry = f"Scaled columns {scale_cols} using '{scale_method}' normalization."
                    st.success(log_entry)
                else:
                    st.warning("Please select columns to scale.")
                    
        with tab4:
            st.subheader("Cleaning Action History Log")
            cleaning_log = projects[active_proj].get("cleaning_log", [])
            if not cleaning_log:
                st.write("No cleaning actions applied yet.")
            else:
                for idx, entry in enumerate(cleaning_log):
                    st.write(f"{idx+1}. `{entry}`")
                    
        # If dataset changed, cache the new version and update project metadata
        if log_entry:
            cached_path = utils.save_dataframe_to_cache(updated_df, projects[active_proj]["original_name"])
            projects[active_proj]["file_path"] = cached_path
            projects[active_proj]["cleaning_log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_entry}")
            utils.save_projects_metadata(projects)
            st.session_state.df = updated_df
            st.session_state.ml_results = None # reset ML models on modification
            st.rerun()
            
    # --- Menu 3: Interactive Visuals ---
    elif choice == "📈 Interactive Visuals":
        st.markdown(f"<h1>📈 Visual Analytics Studio: <span class='gradient-title'>{active_proj}</span></h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8;'>Build beautiful, interactive charts on the fly using Plotly.</p>", unsafe_allow_html=True)
        
        col_ctrl, col_chart = st.columns([1, 3])
        
        with col_ctrl:
            st.markdown("<div class='glass-card'><h4>Plot Configuration</h4></div>", unsafe_allow_html=True)
            plot_type = st.selectbox("Plot Type", [
                "Histogram", "Box Plot", "Scatter Plot", "Bar Chart", 
                "Line Chart", "Pie Chart", "Violin Plot", "3D Scatter Plot", 
                "Area Chart", "Sunburst Chart", "Scatter Matrix (SPLOM)", "Correlation Heatmap"
            ])
            
            # Form variables dynamically
            cols = df.columns.tolist()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            x_col = None
            y_col = None
            z_col = None
            color_col = None
            size_col = None
            bins = 30
            path_cols = []
            splom_cols = []
            corr_cols = []
            corr_metric = "pearson"
            
            if plot_type not in ["Sunburst Chart", "Scatter Matrix (SPLOM)", "Correlation Heatmap"]:
                x_col = st.selectbox("X Axis Column", options=cols)
                
            if plot_type in ["Box Plot", "Scatter Plot", "Line Chart", "Violin Plot", "Bar Chart", "Pie Chart", "3D Scatter Plot", "Area Chart"]:
                y_opts = ["None"] + cols if plot_type in ["Bar Chart", "Pie Chart"] else cols
                y_select = st.selectbox("Y Axis Column", options=y_opts)
                y_col = None if y_select == "None" else y_select
                
            if plot_type == "3D Scatter Plot":
                z_col = st.selectbox("Z Axis Column", options=num_cols)
                
            if plot_type in ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart", "Violin Plot", "3D Scatter Plot", "Area Chart", "Scatter Matrix (SPLOM)"]:
                color_col = st.selectbox("Color Legend Grouping (Optional)", options=["None"] + cols)
                color_col = None if color_col == "None" else color_col
                
            if plot_type in ["Scatter Plot", "3D Scatter Plot"]:
                size_select = st.selectbox("Marker Size Column (Optional)", options=["None"] + num_cols)
                size_col = None if size_select == "None" else size_select
                
            if plot_type == "Histogram":
                bins = st.slider("Bins count", min_value=5, max_value=100, value=30)
                
            if plot_type == "Sunburst Chart":
                path_cols = st.multiselect("Hierarchy Path (Categorical columns in order)", options=[c for c in cols if c not in num_cols], default=[cols[1]] if len(cols) > 1 else [])
                y_select = st.selectbox("Values Column (Optional, Numeric)", options=["None"] + num_cols)
                y_col = None if y_select == "None" else y_select
                
            if plot_type == "Scatter Matrix (SPLOM)":
                splom_cols = st.multiselect("Select Dimensions to Plot", options=num_cols, default=num_cols[:4] if len(num_cols) >= 4 else num_cols)
                
            if plot_type == "Correlation Heatmap":
                corr_cols = st.multiselect("Columns to Correlate", options=num_cols, default=num_cols)
                corr_metric = st.selectbox("Correlation Metric", ["pearson", "spearman"])
                
        with col_chart:
            # Build Plotly Figure
            fig = None
            try:
                if plot_type == "Histogram":
                    fig = eda.plot_histogram(df, x_col, color_col, bins)
                elif plot_type == "Box Plot":
                    fig = eda.plot_box(df, y_col=x_col if not y_col else y_col, x_col=x_col if y_col else None, color_col=color_col)
                elif plot_type == "Scatter Plot":
                    if not y_col:
                        st.error("Scatter plot requires a Y-Axis column.")
                    else:
                        fig = eda.plot_scatter(df, x_col, y_col, color_col, size_col)
                elif plot_type == "Bar Chart":
                    fig = eda.plot_bar(df, x_col, y_col, color_col)
                elif plot_type == "Line Chart":
                    if not y_col:
                        st.error("Line chart requires a Y-Axis column.")
                    else:
                        fig = eda.plot_line(df, x_col, y_col, color_col)
                elif plot_type == "Pie Chart":
                    fig = eda.plot_pie(df, names_col=x_col, values_col=y_col)
                elif plot_type == "Violin Plot":
                    fig = eda.plot_violin(df, y_col=x_col if not y_col else y_col, x_col=x_col if y_col else None, color_col=color_col)
                elif plot_type == "3D Scatter Plot":
                    if not y_col or not z_col:
                        st.error("3D Scatter plot requires both Y-Axis and Z-Axis columns.")
                    else:
                        fig = eda.plot_3d_scatter(df, x_col, y_col, z_col, color_col, size_col)
                elif plot_type == "Area Chart":
                    if not y_col:
                        st.error("Area chart requires a Y-Axis column.")
                    else:
                        fig = eda.plot_area(df, x_col, y_col, color_col)
                elif plot_type == "Sunburst Chart":
                    if not path_cols:
                        st.warning("Please select at least one hierarchy column.")
                    else:
                        fig = eda.plot_sunburst(df, path_cols, y_col)
                elif plot_type == "Scatter Matrix (SPLOM)":
                    if not splom_cols:
                        st.warning("Please select at least one dimension.")
                    else:
                        fig = eda.plot_splom(df, splom_cols, color_col)
                elif plot_type == "Correlation Heatmap":
                    if not corr_cols:
                        st.warning("Please select at least one column to correlate.")
                    else:
                        corr_matrix = eda.calculate_correlations(df[corr_cols], method=corr_metric)
                        fig = eda.plot_heatmap(corr_matrix)
            except Exception as e:
                st.error(f"Error plotting chart: {e}")
                
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
    # --- Menu 4: Statistical Models ---
    elif choice == "🧮 Statistical Models":
        st.markdown(f"<h1>🧮 Statistical Hypothesis Hub: <span class='gradient-title'>{active_proj}</span></h1>", unsafe_allow_html=True)
        
        stat_choice = st.selectbox("Statistical Method", ["Correlation Analysis", "Two-Sample T-Test", "ANOVA (Analysis of Variance)", "OLS Linear Regression"])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if stat_choice == "Correlation Analysis":
            st.subheader("Correlation Analysis")
            if len(numeric_cols) < 2:
                st.error("Require at least 2 numerical columns.")
            else:
                c_method = st.selectbox("Correlation Metric", ["pearson", "spearman"])
                corr_matrix = eda.calculate_correlations(df, method=c_method)
                st.dataframe(corr_matrix, use_container_width=True)
                
        elif stat_choice == "Two-Sample T-Test":
            st.subheader("Two-Sample T-Test (Independent)")
            if not numeric_cols or not categorical_cols:
                st.error("Requires at least one numerical variable and one categorical grouping variable.")
            else:
                num_var = st.selectbox("Numerical Variable", numeric_cols)
                cat_var = st.selectbox("Grouping Variable (Categorical)", categorical_cols)
                
                # Check for categories count
                cats = df[cat_var].dropna().unique()
                if len(cats) != 2:
                    st.warning(f"T-test requires exactly 2 grouping values in your category. Variable '{cat_var}' has {len(cats)} values.")
                    st.write("Unique classes found:", cats)
                else:
                    from scipy import stats
                    group1 = df[df[cat_var] == cats[0]][num_var].dropna()
                    group2 = df[df[cat_var] == cats[1]][num_var].dropna()
                    
                    if len(group1) < 2 or len(group2) < 2:
                        st.error("Insufficient samples in groups to run T-test.")
                    else:
                        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                        st.write(f"**Group 1 ({cats[0]}):** Mean = `{group1.mean():.4f}`, N = `{len(group1)}`")
                        st.write(f"**Group 2 ({cats[1]}):** Mean = `{group2.mean():.4f}`, N = `{len(group2)}`")
                        st.write(f"**T-Statistic:** `{t_stat:.4f}`")
                        st.write(f"**P-Value:** `{p_val:.4g}`")
                        
                        if p_val < 0.05:
                            st.success(f"**Significant Result (p < 0.05):** Reject the null hypothesis. There is a statistically significant difference in '{num_var}' mean values across the groups.")
                        else:
                            st.info(f"**Non-Significant Result (p >= 0.05):** Fail to reject the null hypothesis. There is no statistically significant difference in '{num_var}' mean values between the two groups.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
        elif stat_choice == "ANOVA (Analysis of Variance)":
            st.subheader("One-Way ANOVA")
            if not numeric_cols or not categorical_cols:
                st.error("Requires at least one numerical variable and one categorical grouping variable.")
            else:
                num_var = st.selectbox("Numerical Variable", numeric_cols)
                cat_var = st.selectbox("Grouping Variable (Categorical)", categorical_cols)
                
                cats = df[cat_var].dropna().unique()
                if len(cats) < 2:
                    st.error("Categorical grouping column must have at least 2 distinct groups.")
                else:
                    from scipy import stats
                    groups_data = [df[df[cat_var] == cat][num_var].dropna() for cat in cats]
                    
                    # Filter empty groups
                    groups_data = [g for g in groups_data if len(g) > 1]
                    if len(groups_data) < 2:
                        st.error("Insufficient samples inside categorical groups to compute ANOVA.")
                    else:
                        f_stat, p_val = stats.f_oneway(*groups_data)
                        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                        st.write(f"**F-Statistic:** `{f_stat:.4f}`")
                        st.write(f"**P-Value:** `{p_val:.4g}`")
                        
                        if p_val < 0.05:
                            st.success(f"**Significant Result (p < 0.05):** The means of at least one group differ significantly from the others.")
                        else:
                            st.info(f"**Non-Significant Result (p >= 0.05):** No statistically significant difference detected among group means.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
        elif stat_choice == "OLS Linear Regression":
            st.subheader("Ordinary Least Squares Linear Regression")
            if len(numeric_cols) < 2:
                st.error("Requires at least 2 numerical columns (1 Target, 1 or more Predictors).")
            else:
                target_reg = st.selectbox("Target Column (Y)", numeric_cols)
                predictors = st.multiselect("Predictors (X)", [x for x in numeric_cols if x != target_reg])
                
                if st.button("Fit Model") and predictors:
                    import statsmodels.api as sm
                    df_reg = df[[target_reg] + predictors].dropna()
                    
                    Y_data = df_reg[target_reg]
                    X_data = df_reg[predictors]
                    X_data = sm.add_constant(X_data)
                    
                    try:
                        model = sm.OLS(Y_data, X_data).fit()
                        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                        st.write(f"**R-squared:** `{model.rsquared:.4f}`")
                        st.write(f"**Adjusted R-squared:** `{model.rsquared_adj:.4f}`")
                        st.write(f"**F-statistic p-value:** `{model.f_pvalue:.4g}`")
                        st.dataframe(model.summary2().tables[1], use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Plot residuals distribution
                        residuals = model.resid
                        fig_res = px.histogram(residuals, title="Residuals Distribution", labels={'value': 'Residual Value'}, template="plotly_dark")
                        st.plotly_chart(fig_res, use_container_width=True)
                    except Exception as e:
                        st.error(f"Regression error: {e}")
                        
    # --- Menu 5: AutoML Hub ---
    elif choice == "🤖 AutoML Hub":
        st.markdown(f"<h1>🤖 AutoML Hub: <span class='gradient-title'>{active_proj}</span></h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8;'>Train and evaluate multiple models concurrently on your dataset with auto-preprocessing.</p>", unsafe_allow_html=True)
        
        target_col = st.selectbox("Select Target Column to Predict", options=df.columns)
        
        # Select features
        available_features = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect("Select Feature Columns", options=available_features, default=available_features)
        
        task_type = st.radio("Task Type Selection", ["Auto-Detect", "Classification", "Regression"])
        if task_type == "Auto-Detect":
            detected = ml.auto_detect_task_type(df, target_col)
            st.info(f"Auto-detected task type: **{detected.capitalize()}**")
            active_task_type = detected
        else:
            active_task_type = task_type.lower()
            
        test_pct = st.slider("Train/Test Split (Test Size %)", min_value=10, max_value=50, value=20)
        
        if st.button("🚀 Run AutoML Models"):
            if not selected_features:
                st.error("Please select at least one feature column.")
            else:
                with st.spinner("Preprocessing dataset, training models, and generating metrics..."):
                    try:
                        X, y, feat_names, meta = ml.preprocess_for_ml(df, target_col, selected_features)
                        results = ml.train_and_evaluate(X, y, active_task_type, test_size=(test_pct/100.0))
                        
                        # Store trained results inside session state
                        st.session_state.ml_results = {
                            "results": results,
                            "feat_names": feat_names,
                            "task_type": active_task_type,
                            "target_col": target_col,
                            "meta": meta
                        }
                        st.success("AutoML training complete!")
                    except Exception as e:
                        st.error(f"ML Processing error: {e}")
                        
        if st.session_state.ml_results:
            ml_data = st.session_state.ml_results
            results = ml_data["results"]
            feat_names = ml_data["feat_names"]
            active_task_type = ml_data["task_type"]
            target_col = ml_data["target_col"]
            
            st.markdown("<div class='glass-card'><h3>🏆 Model Leaderboard</h3></div>", unsafe_allow_html=True)
            
            # Format leaderboard metrics into table
            leaderboard_rows = []
            for name, res in results.items():
                if "error" in res:
                    continue
                row = {"Model Name": name}
                row.update(res["metrics"])
                leaderboard_rows.append(row)
                
            lead_df = pd.DataFrame(leaderboard_rows)
            st.dataframe(lead_df, use_container_width=True, hide_index=True)
            
            # Detail view of the best model
            best_model_name = ""
            best_metric = -999.0
            metric_to_compare = "Accuracy" if active_task_type == "classification" else "R2-Score (R-squared)"
            
            for name, res in results.items():
                if "error" in res:
                    continue
                val = res["metrics"][metric_to_compare]
                if val > best_metric:
                    best_metric = val
                    best_model_name = name
                    
            st.success(f"Best performing model by **{metric_to_compare}**: **{best_model_name}** (`{best_metric:.4f}`)")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown(f"<div class='glass-card'><h3>📊 Feature Importances ({best_model_name})</h3></div>", unsafe_allow_html=True)
                importances = results[best_model_name].get("feature_importances", [])
                if importances and len(importances) == len(feat_names):
                    imp_df = pd.DataFrame({
                        "Feature": feat_names,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False).head(15)
                    
                    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h', title="Feature Importance Coefficients", template="plotly_dark")
                    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Feature importance metrics unavailable for this model.")
                    
            with col_right:
                if active_task_type == "classification":
                    st.markdown(f"<div class='glass-card'><h3>🧩 Confusion Matrix ({best_model_name})</h3></div>", unsafe_allow_html=True)
                    cm = results[best_model_name].get("confusion_matrix", [])
                    if cm:
                        class_labels = ml_data["meta"].get("target_classes")
                        if not class_labels:
                            class_labels = [str(x) for x in range(len(cm))]
                        
                        fig_cm = px.imshow(
                            cm, 
                            text_auto=True, 
                            aspect="auto", 
                            labels=dict(x="Predicted", y="Actual"),
                            x=class_labels,
                            y=class_labels,
                            template="plotly_dark",
                            color_continuous_scale="Blues"
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.markdown("<div class='glass-card'><h3>📉 Model Residuals</h3></div>", unsafe_allow_html=True)
                    st.info("Continuous regression target does not generate a categorical confusion matrix.")
                    
    # --- Menu 6: Summaries & Reports ---
    elif choice == "📝 Summaries & Reports":
        st.markdown(f"<h1>📝 Executive Summary Builder: <span class='gradient-title'>{active_proj}</span></h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8;'>Write notes, summarize findings, and download a professional PDF summary report.</p>", unsafe_allow_html=True)
        
        st.subheader("Add Analyst Interpretations")
        user_notes = st.text_area("Observations, insights, and recommendations", value=st.session_state.notes, height=200)
        st.session_state.notes = user_notes
        
        if st.button("Generate & Download PDF Report"):
            summary_stats = eda.get_dataset_summary(df)
            col_metadata_df = eda.get_columns_metadata(df)
            ml_results = st.session_state.ml_results.get("results", {}) if st.session_state.ml_results else {}
            
            with st.spinner("Compiling PDF document..."):
                try:
                    pdf_filepath = utils.generate_pdf_report(active_proj, summary_stats, col_metadata_df, ml_results, user_notes)
                    st.success("Report successfully generated!")
                    
                    with open(pdf_filepath, "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                        
                    st.download_button(
                        label="Download PDF Report 📄",
                        data=pdf_data,
                        file_name=os.path.basename(pdf_filepath),
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")
                    
    # --- Menu 7: AI Assistant ---
    elif choice == "💬 AI Assistant":
        st.markdown(f"<h1>💬 Smart Analyst Assistant: <span class='gradient-title'>{active_proj}</span></h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8;'>Ask logical questions about your dataset. The assistant evaluates metrics dynamically.</p>", unsafe_allow_html=True)
        
        query = st.text_input("Ask a question about the active dataset (e.g., 'Summarize columns', 'Show columns with missing values', 'Tell me target predictions')")
        
        if query:
            query_l = query.lower()
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.write(f"💬 **Question:** {query}")
            st.write("---")
            
            # Dynamic rules-based response builder mimicking AI analytics
            if "missing" in query_l or "null" in query_l:
                missing_df = df.isna().sum().reset_index()
                missing_df.columns = ["Column", "Missing Count"]
                missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False)
                
                if missing_df.empty:
                    st.success("The dataset contains zero missing values!")
                else:
                    st.write("Here are the columns containing missing values in descending order:")
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)
            elif "summarize" in query_l or "summary" in query_l:
                summary = eda.get_dataset_summary(df)
                st.write(f"This dataset has **{summary['num_rows']}** rows and **{summary['num_cols']}** columns.")
                st.write(f"- Numerical features count: `{summary['numeric_cols']}`")
                st.write(f"- Categorical features count: `{summary['categorical_cols']}`")
                st.write(f"- Total missing cell percentage: `{summary['missing_percentage']:.2f}%`")
                st.write(f"- Total duplicates: `{summary['duplicate_rows']}` rows.")
            elif "correlation" in query_l:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) < 2:
                    st.write("Insufficient numerical columns to run correlation checks.")
                else:
                    # Find highest absolute correlation pair
                    corr = df[numeric_cols].corr().abs().stack().reset_index()
                    corr.columns = ['V1', 'V2', 'R']
                    corr = corr[corr['V1'] != corr['V2']].sort_values(by='R', ascending=False)
                    if not corr.empty:
                        best = corr.iloc[0]
                        act_val = df[best['V1']].corr(df[best['V2']])
                        st.write(f"The highest correlated variables are **{best['V1']}** and **{best['V2']}** with a Pearson correlation index of `{act_val:.4f}`.")
                    else:
                        st.write("Could not compute correlations.")
            elif "predict" in query_l:
                if st.session_state.ml_results:
                    ml_data = st.session_state.ml_results
                    st.write(f"An AutoML model pipeline has already been run predicting target variable **{ml_data['target_col']}**.")
                    st.write(f"- AutoML Model Type: `{ml_data['task_type'].capitalize()}`")
                    st.write("- Model Leaders:")
                    for k, v in ml_data["results"].items():
                        if "error" not in v:
                            st.write(f"  * **{k}**: {v['metrics']}")
                else:
                    st.write("No machine learning models have been trained yet in this session. Head to the 'AutoML Hub' to configure and execute predictors first.")
            else:
                # Default generic fallback
                st.write("I understand your query, but my statistical parsing module requires more details. Try asking:")
                st.markdown("""
                - *'Summarize dataset'*
                - *'Which columns contain missing cells?'*
                - *'What is the highest correlation in the dataset?'*
                - *'Tell me target predictions' (if ML model was run)*
                """)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
