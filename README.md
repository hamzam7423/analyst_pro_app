# 📈 Data Analyst Pro – Portfolio Projects

Welcome to my **Data Analyst Pro** app and portfolio!  
This repo showcases how I handle **real-world messy data**: cleaning, analyzing, visualizing, and delivering business insights.  

I built a custom **Streamlit app** for data cleaning and analysis, and applied it to **3 realistic tasks** with raw datasets + manager briefs.  

---

## 🚀 Features of the App
- **Data Cleaning**
  - Manual recipe builder (rename, drop, split, fillna, dedupe).
  - Natural language cleaning (via Ollama, optional).
  - Save/Load JSON cleaning plans.
  - Batch clean multiple files.

- **Analysis**
  - Interactive filters (category + date).
  - Pivot builder with aggregations.
  - Time-series resampling (daily/weekly/monthly).
  - Data profiling: missing values, numeric summaries.

- **Visualization**
  - Bar charts, line charts, histograms (matplotlib).

- **Export**
  - Clean CSV and Parquet.
  - BI-ready folder for Power BI/Tableau.
  - Executive Summary (Markdown).

---

## 📂 Portfolio Tasks

Each task simulates a **manager request**, with a raw dataset and a clear brief.  
I used the app to clean the data, generate insights, and deliver outputs.

### 🔹 [Task 1: Company Earnings](tasks/task1_manager_brief.md)
- Clean messy financial data (mixed dates, inconsistent company names, missing revenues).
- Deliverables:
  - Clean CSV + Parquet
  - Summary report
  - Revenue by company + monthly trend charts

Dataset → [`company_earnings_raw.csv`](data/company_earnings_raw.csv)

---

### 🔹 [Task 2: Employee Records](tasks/task2_manager_brief.md)
- Clean HR data (split full names, standardize DOBs/emails, handle missing departments and salaries).
- Deliverables:
  - Clean CSV + Parquet
  - Summary report
  - Avg salary per department chart

Dataset → [`employee_records_raw.csv`](data/employee_records_raw.csv)

---

### 🔹 [Task 3: Sales Orders](tasks/task3_manager_brief.md)
- Clean sales data (fix quantities, standardize dates, calculate totals).
- Deliverables:
  - Clean CSV + Parquet
  - Summary report
  - Revenue by product + monthly sales trend charts

Dataset → [`sales_orders_raw.csv`](data/sales_orders_raw.csv)

---

## 🖥️ How to Run the App

1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/analyst_pro_app.git
   cd analyst_pro_app
