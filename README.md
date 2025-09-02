# analyst_pro_app

# Data Analyst Pro (Local App)

A privacy-friendly **data analyst workbench** you can run on your laptop.
Clean messy CSV/XLSX with natural language (optional **Ollama**), explore, build pivots,
chart results, and export BI-ready datasets (CSV/Parquet) for Power BI/Tableau.

## Features
- **Cleaning**
  - Natural-language plans via **Ollama** (optional, local).
  - Manual **Recipe Builder** (rename, parse dates, split, fillna, dedupe, drop).
  - **Save/Load cleaning plans** (JSON) and **Batch apply** to many files.
- **Analysis**
  - **Interactive filters** (category/date).
  - **Pivot builder** (group by, sum/avg/count/median/max/min, % of total).
  - **Time-series resampling** (D/W/M).
- **Visualization**
  - Bar / Time-series line / Histogram (matplotlib).
- **Quality & Reporting**
  - Missingness, numeric summary, quick outlier check.
  - **Executive Summary** (Markdown) download.
- **Export**
  - **CSV** and **Parquet**.
  - One-click **BI folder** (zip) with `/data/cleaned.csv`, `/data/cleaned.parquet`, `/meta/profile.json`.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
