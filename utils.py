import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
PROJECTS_FILE = os.path.join(CACHE_DIR, "projects.json")

def init_cache():
    """Ensure the cache directories exist."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    if not os.path.exists(PROJECTS_FILE):
        with open(PROJECTS_FILE, 'w') as f:
            json.dump({}, f)

def load_projects_metadata() -> dict:
    """Load metadata of all projects."""
    init_cache()
    try:
        with open(PROJECTS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_projects_metadata(metadata: dict):
    """Save metadata of all projects."""
    init_cache()
    with open(PROJECTS_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from path matching the extension."""
    _, ext = os.path.splitext(file_path.lower())
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    elif ext == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def save_dataframe_to_cache(df: pd.DataFrame, filename: str) -> str:
    """Save a DataFrame as Parquet in the cache and return the file path."""
    init_cache()
    base_name, _ = os.path.splitext(filename)
    cached_filename = f"{base_name}_{int(datetime.now().timestamp())}.parquet"
    target_path = os.path.join(CACHE_DIR, cached_filename)
    df.to_parquet(target_path, index=False)
    return target_path

def generate_pdf_report(project_name: str, summary_stats: dict, col_metadata_df: pd.DataFrame, ml_results: dict, notes: str = "") -> str:
    """Generate a clean PDF report summarizing dataset stats and ML model results."""
    init_cache()
    filename = f"report_{project_name.lower().replace(' ', '_')}_{int(datetime.now().timestamp())}.pdf"
    pdf_path = os.path.join(CACHE_DIR, filename)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=15
    )
    subtitle_style = ParagraphStyle(
        'ReportSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=25
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=15,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#334155'),
        spaceAfter=10,
        leading=14
    )
    
    story = []
    
    # Title & Metadata
    story.append(Paragraph(f"Data Analytics Summary Report: {project_name}", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Dataset summary section
    story.append(Paragraph("1. Dataset Overview", heading_style))
    sum_text = (
        f"<b>Total Rows:</b> {summary_stats.get('num_rows', 0)}<br/>"
        f"<b>Total Columns:</b> {summary_stats.get('num_cols', 0)}<br/>"
        f"<b>Numeric Columns:</b> {summary_stats.get('numeric_cols', 0)}<br/>"
        f"<b>Categorical Columns:</b> {summary_stats.get('categorical_cols', 0)}<br/>"
        f"<b>Missing Cells Count:</b> {summary_stats.get('missing_cells', 0)} ({summary_stats.get('missing_percentage', 0.0):.2f}%)<br/>"
        f"<b>Duplicate Rows:</b> {summary_stats.get('duplicate_rows', 0)}"
    )
    story.append(Paragraph(sum_text, body_style))
    story.append(Spacer(1, 10))
    
    # Table of Columns
    story.append(Paragraph("2. Column Specifications", heading_style))
    col_data = [["Column Name", "Type", "Missing %", "Unique Values"]]
    for _, row in col_metadata_df.head(15).iterrows():
        col_data.append([
            str(row["Column Name"]),
            str(row["Data Type"]),
            f"{row['Missing %']}%",
            str(row["Unique Values"])
        ])
    
    col_table = Table(col_data, hAlign='LEFT')
    col_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(col_table)
    if len(col_metadata_df) > 15:
        story.append(Paragraph(f"* Showing top 15 columns out of {len(col_metadata_df)} total columns.", body_style))
    story.append(Spacer(1, 15))
    
    # ML Section (if present)
    if ml_results:
        story.append(Paragraph("3. Machine Learning Model Performance", heading_style))
        story.append(Paragraph("Autonomously evaluated models returned the following metrics:", body_style))
        
        ml_data = [["Model Name", "Metric", "Value"]]
        for model_name, model_info in ml_results.items():
            if "error" in model_info:
                ml_data.append([model_name, "Error", model_info["error"][:30]])
                continue
            for metric_name, val in model_info.get("metrics", {}).items():
                ml_data.append([model_name, metric_name, f"{val:.4f}"])
        
        ml_table = Table(ml_data, hAlign='LEFT')
        ml_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2563EB')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#EFF6FF')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#BFDBFE')),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(ml_table)
        story.append(Spacer(1, 15))
        
    # Custom Analyst Notes
    if notes:
        story.append(Paragraph("4. Executive Summary & Analyst Notes", heading_style))
        story.append(Paragraph(notes.replace("\n", "<br/>"), body_style))
        
    # Build Document
    doc.build(story)
    return pdf_path
