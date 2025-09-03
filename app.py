import io, os, json, requests, streamlit as st, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from dateutil import parser as dtparser
from datetime import datetime
from typing import List, Dict

st.set_page_config(page_title="Data Analyst Pro", page_icon="📈", layout="wide")

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """You are a data cleaning planner.
Given a short natural-language description of how a user wants to clean a tabular dataset,
produce a STRICT JSON plan using this schema:

{
  "rename": {"old_name":"new_name"},
  "drop_columns": ["colA","colB"],
  "standardize_case": {"columns":["name"], "mode":"title|lower|upper"},
  "parse_dates": {"columns":["date","dob"], "format":"infer|%Y-%m-%d"},
  "split": [{"column":"full_name","into":["first_name","last_name"], "delimiter":" " }],
  "fillna": {"column_defaults": {"age": 0, "email": ""}},
  "deduplicate": {"subset":["email"], "keep":"first|last"},
  "separate_headings": {"from_column":"notes","targets":["issue","detail"], "delimiter":";"},
  "filters": [{"column":"status","op":"in","value":["active","pending"]}]  // optional
}

Rules:
- Return ONLY valid JSON. No prose, no markdown.
- Include only operations that were requested or are clearly implied.
- Use exact column names from the dataset when relevant; otherwise omit.
"""

# ---------- Helpers

def call_ollama_get_plan(instructions: str, columns_hint=None):
    user_prompt = "Instructions: " + instructions
    if columns_hint is not None:
        user_prompt += "\n\nAvailable columns: " + ", ".join(columns_hint)

    payload = {
        "model": "mistral",
        "prompt": f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{user_prompt}\n\nAssistant:",
        "stream": False,
        "options": {"temperature": 0.2}
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        return json.loads(text)
    except Exception as e:
        st.info(f"Could not use Ollama planner (falling back to manual). Reason: {e}")
        return None

def safe_standardize_case(df, cols, mode):
    for c in cols:
        if c in df.columns and df[c].dtype == object:
            s = df[c].astype(str).str.strip()
            if mode == "lower":
                df[c] = s.str.lower()
            elif mode == "upper":
                df[c] = s.str.upper()
            elif mode == "title":
                df[c] = s.str.title()
    return df

def safe_parse_dates(df, cols, fmt):
    for c in cols:
        if c in df.columns:
            if fmt == "infer":
                df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            else:
                df[c] = pd.to_datetime(df[c], errors="coerce", format=fmt)
    return df

def safe_split(df, column, into, delimiter):
    if column in df.columns and len(into) >= 2:
        parts = df[column].astype(str).str.split(delimiter, n=len(into)-1, expand=True)
        for i, name in enumerate(into):
            df[name] = parts[i] if i in parts.columns else None
    return df

def apply_filters(df, filters: List[Dict]):
    if not filters: return df
    res = df.copy()
    for f in filters:
        col = f.get("column"); op = f.get("op"); val = f.get("value")
        if col not in res.columns: continue
        if op == "in" and isinstance(val, list):
            res = res[res[col].isin(val)]
        elif op == "not_in" and isinstance(val, list):
            res = res[~res[col].isin(val)]
        elif op == "equals":
            res = res[res[col] == val]
        elif op == "not_equals":
            res = res[res[col] != val]
        elif op == "gt":
            res = res[pd.to_numeric(res[col], errors="coerce") > float(val)]
        elif op == "gte":
            res = res[pd.to_numeric(res[col], errors="coerce") >= float(val)]
        elif op == "lt":
            res = res[pd.to_numeric(res[col], errors="coerce") < float(val)]
        elif op == "lte":
            res = res[pd.to_numeric(res[col], errors="coerce") <= float(val)]
    return res

def apply_plan(df: pd.DataFrame, plan: dict):
    if not plan:
        return df

    if "rename" in plan and isinstance(plan["rename"], dict):
        df = df.rename(columns=plan["rename"])

    if "drop_columns" in plan:
        drops = [c for c in plan["drop_columns"] if c in df.columns]
        if drops:
            df = df.drop(columns=drops)

    if "standardize_case" in plan:
        sc = plan["standardize_case"]
        cols = sc.get("columns", [])
        mode = sc.get("mode", "lower")
        df = safe_standardize_case(df, cols, mode)

    if "parse_dates" in plan:
        pdx = plan["parse_dates"]
        cols = pdx.get("columns", [])
        fmt = pdx.get("format", "infer")
        df = safe_parse_dates(df, cols, fmt)

    if "split" in plan and isinstance(plan["split"], list):
        for spec in plan["split"]:
            col = spec.get("column")
            into = spec.get("into", [])
            delim = spec.get("delimiter", " ")
            df = safe_split(df, col, into, delim)

    if "fillna" in plan and isinstance(plan["fillna"], dict):
        cd = plan["fillna"].get("column_defaults", {})
        for col, val in cd.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

  if "deduplicate" in plan and isinstance(plan["deduplicate"], dict):
    subset = plan["deduplicate"].get("subset", None)
    keep = plan["deduplicate"].get("keep", "first")

    # ✅ Remove empty strings from the subset
    if subset:
        subset = [col for col in subset if col.strip() != '']

    # ✅ Proceed only if subset is not empty and valid
    if subset:
        keep = keep if keep in ("first", "last") else "first"
        if all(col in df.columns for col in subset):
            df = df.drop_duplicates(subset=subset, keep=keep)
        else:
            st.warning("⚠️ Some columns in the deduplication plan don't exist in the DataFrame. Skipping deduplication.")
    else:
        df = df.drop_duplicates()


    if "filters" in plan and isinstance(plan["filters"], list):
        df = apply_filters(df, plan["filters"])

    if "separate_headings" in plan:
        spec = plan["separate_headings"]
        df = safe_split(df, spec.get("from_column"), spec.get("targets", []), spec.get("delimiter", ";"))

    #Auto-fi: force numeric for common financial columns
    for c in df.columns:
        if any(k in c.lower() for k in ["revenue", "amount", "price", "total"]):
            df[c] = pd.to_numeric(df[c], errors="coerce"),fillna(0)

    return df

def default_clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    df = df.drop_duplicates()
    datey = [c for c in df.columns if "date" in c.lower() or "dob" in c.lower()]
    df = safe_parse_dates(df, datey, "infer")
    return df

def numeric_columns(df): return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
def categorical_columns(df): return [c for c in df.columns if df[c].dtype == object or pd.api.types.is_categorical_dtype(df[c])]
def datetime_columns(df): return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

# ---------- UI

st.title("📈 Data Analyst Pro (Local App)")
st.caption("Clean → Analyze → Visualize → Export (BI-ready)")

with st.sidebar:
    st.header("1) Upload")
    file = st.file_uploader("CSV or Excel", type=["csv","xlsx"])
    multi_files = st.file_uploader("Batch files (optional)", type=["csv","xlsx"], accept_multiple_files=True)
    st.header("2) LLM Planning (optional)")
    use_llm = st.checkbox("Use local LLM via Ollama", value=True)
    instructions = st.text_area("Describe cleaning you'd like")
    st.header("3) Plans")
    plan_file = st.file_uploader("Load plan (JSON)", type=["json"])
    plan_save_name = st.text_input("Save plan as", value="cleaning_plan.json")
    save_plan_clicked = st.button("Save current plan")

tab_clean, tab_analyze, tab_chart, tab_export = st.tabs(["🧹 Clean", "🔎 Analyze", "📊 Charts", "📤 Export"])

if "plan" not in st.session_state: st.session_state.plan = None
if "cleaned" not in st.session_state: st.session_state.cleaned = None
if "orig" not in st.session_state: st.session_state.orig = None

def load_df(uploaded):
    if uploaded is None: return None
    try:
        if uploaded.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded)
        else:
            return pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

orig = load_df(file)
if orig is not None:
    st.session_state.orig = orig

if plan_file is not None:
    try:
        st.session_state.plan = json.load(plan_file)
        st.success("Loaded plan from file.")
    except Exception as e:
        st.error(f"Invalid plan JSON: {e}")

if save_plan_clicked:
    if st.session_state.plan:
        st.download_button("Download plan JSON", data=json.dumps(st.session_state.plan, indent=2),
                           file_name=plan_save_name, mime="application/json", key="dlplan")

# ---- CLEAN TAB
with tab_clean:
    st.subheader("Preview")
    if st.session_state.orig is not None:
        st.dataframe(st.session_state.orig.head(50))
    else:
        st.info("Upload a file to begin.")

    plan = None
    if st.session_state.orig is not None and use_llm and instructions.strip():
        with st.spinner("Asking local LLM for a cleaning plan..."):
            plan = call_ollama_get_plan(instructions, columns_hint=list(st.session_state.orig.columns))

    st.markdown("### Manual Recipe (optional)")
    with st.expander("Build a quick recipe"):
        rename_pairs = st.text_area("Rename (old:new per line)", placeholder="cust id:customer_id\n Date :date")
        drop_cols = st.text_input("Drop columns (comma-separated)")
        std_mode = st.selectbox("Standardize case", ["skip","lower","upper","title"], index=0)
        std_cols = st.text_input("Columns to standardize (comma-separated)")
        parse_cols = st.text_input("Parse dates (comma-separated)")
        parse_fmt = st.text_input("Date format (e.g., %Y-%m-%d or infer)", value="infer")
        fill_defaults = st.text_area("Fill missing defaults (col=value per line)", placeholder="age=0\nemail=")
        dedup_subset = st.text_input("Deduplicate subset (comma-separated)")
        dedup_keep = st.selectbox("Deduplicate keep", ["first","last"])

        manual = {}
        if rename_pairs.strip():
            ren = {}
            for line in rename_pairs.splitlines():
                if ":" in line:
                    a,b = line.split(":",1); ren[a.strip()] = b.strip()
            if ren: manual["rename"] = ren
        if drop_cols.strip():
            manual["drop_columns"] = [c.strip() for c in drop_cols.split(",") if c.strip()]
        if std_mode != "skip" and std_cols.strip():
            manual["standardize_case"] = {"columns": [c.strip() for c in std_cols.split(",")], "mode": std_mode}
        if parse_cols.strip():
            manual["parse_dates"] = {"columns": [c.strip() for c in parse_cols.split(",")], "format": parse_fmt.strip() or "infer"}
        if fill_defaults.strip():
            cd = {}
            for line in fill_defaults.splitlines():
                if "=" in line:
                    c,v = line.split("=",1); cd[c.strip()] = v
            manual["fillna"] = {"column_defaults": cd}
        if dedup_subset.strip():
            manual["deduplicate"] = {"subset": [c.strip() for c in dedup_subset.split(",")], "keep": dedup_keep}

    if plan and manual:
        plan.update(manual)
    elif manual:
        plan = manual

    if plan:
        st.session_state.plan = plan
        st.markdown("**Current Plan**")
        st.code(json.dumps(plan, indent=2), language="json")

    if st.button("Apply cleaning"):
        if st.session_state.orig is not None:
            cleaned = apply_plan(st.session_state.orig.copy(), st.session_state.plan) if st.session_state.plan else default_clean(st.session_state.orig.copy())
            st.session_state.cleaned = cleaned
            st.success("Cleaning applied.")
        else:
            st.warning("Upload a file first.")

    if st.session_state.cleaned is not None:
        st.subheader("Cleaned preview")
        st.dataframe(st.session_state.cleaned.head(50))

    st.markdown("---")
    st.subheader("Batch apply plan to multiple files")
    multi_files = st.session_state.get("multi_files", None)
    # (Handled via sidebar uploader in practice—download buttons produced per file after apply.)

# ---- ANALYZE TAB
with tab_analyze:
    st.subheader("Data profiling & filters")
    if st.session_state.cleaned is None:
        st.info("Apply cleaning first.")
    else:
        df = st.session_state.cleaned.copy()
        with st.expander("Filters"):
            cats = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_categorical_dtype(df[c])]
            nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            dts  = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

            for c in cats[:5]:
                opts = ["<All>"] + sorted(df[c].astype(str).dropna().unique().tolist())[:1000]
                choice = st.selectbox(f"{c}", opts, key=f"f_{c}")
                if choice != "<All>":
                    df = df[df[c].astype(str) == choice]

            if dts:
                dtc = st.selectbox("Date column (optional)", ["<None>"] + dts, index=0)
                if dtc != "<None>":
                    min_d, max_d = df[dtc].min(), df[dtc].max()
                    dr = st.slider("Date range",
                                   min_value=min_d.to_pydatetime() if hasattr(min_d, 'to_pydatetime') else min_d,
                                   max_value=max_d.to_pydatetime() if hasattr(max_d, 'to_pydatetime') else max_d,
                                   value=(min_d, max_d))
                    df = df[(df[dtc] >= dr[0]) & (df[dtc] <= dr[1])]

        st.markdown("### Summary")
        st.write(f"Rows: {len(df)}  |  Columns: {len(df.columns)}")
        with st.expander("Numeric describe"):
            import numpy as np  # Make sure this is at the top of your script

# Step-by-step handling
if df.empty:
    st.warning("⚠️ The DataFrame is empty. Nothing to describe.")
else:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.warning("⚠️ No numeric data available to describe.")
    else:
        st.dataframe(numeric_df.describe().transpose())

        with st.expander("Missing values by column"):
            miss = df.isna().sum().to_frame("missing_count")
            miss["missing_pct"] = (miss["missing_count"] / len(df) * 100).round(2)
            st.dataframe(miss.sort_values("missing_count", ascending=False))

        st.markdown("### Pivot builder")
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        group_col = st.selectbox("Group by (category)", ["<None>"] + [c for c in df.columns if c not in nums], index=0)
        if group_col != "<None>" and nums:
            metric = st.selectbox("Metric", nums)
            agg = st.selectbox("Aggregation", ["sum","mean","median","count","max","min"], index=0)
            pv = df.groupby(group_col)[metric].agg(agg).reset_index().sort_values(metric, ascending=False)
            st.dataframe(pv.head(100))
            pot = (pv[metric] / pv[metric].sum() * 100).round(2)
            st.write("Percent of total (top rows):")
            st.dataframe(pd.DataFrame({group_col: pv[group_col], f"{metric}_{agg}": pv[metric], "pct_total": pot}).head(100))

        dts  = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dts and nums:
            st.markdown("### Time series resample")
            dtcol = st.selectbox("Date/Time column", dts, key="ts_dt")
            ts_metric = st.selectbox("Metric", nums, key="ts_metric")
            freq = st.selectbox("Frequency", ["D","W","M"], index=2)
            method = st.selectbox("Aggregation", ["sum","mean","median","count","max","min"], index=0)
            ts = df[[dtcol, ts_metric]].dropna().set_index(dtcol).sort_index()
            if method == "sum":
                res = ts.resample(freq).sum()
            elif method == "mean":
                res = ts.resample(freq).mean()
            elif method == "median":
                res = ts.resample(freq).median()
            elif method == "count":
                res = ts.resample(freq).count()
            elif method == "max":
                res = ts.resample(freq).max()
            else:
                res = ts.resample(freq).min()
            st.dataframe(res.head(100))

# ---- CHARTS TAB
with tab_chart:
    st.subheader("Chart gallery")
    if st.session_state.cleaned is None:
        st.info("Apply cleaning first.")
    else:
        df = st.session_state.cleaned.copy()
      #Debug block to show column name & types
      with st.expander("columns & types (debug)"):
          st.write(df.dtypes.astype(str))
          st.dataframe(df.head(10))
        
        cats = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_categorical_dtype(df[c])]
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        dts  = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        chart = st.selectbox("Chart type", ["Bar", "Line (time)", "Histogram"], index=0)

        if chart == "Bar" and cats and nums:
            xcol = st.selectbox("Category (x)", cats)
            ycol = st.selectbox("Metric (y)", nums)
            agg = st.selectbox("Aggregation", ["sum","mean","median","count","max","min"], index=0)
            grouped = df.groupby(xcol)[ycol].agg(agg).reset_index().sort_values(ycol, ascending=False).head(50)
            fig = plt.figure()
            plt.bar(grouped[xcol].astype(str).tolist(), grouped[ycol].tolist())
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{agg} of {ycol} by {xcol}")
            st.pyplot(fig)

        if chart == "Line (time)" and dts and nums:
            dtcol = st.selectbox("Date/Time column", dts, key="line_dt")
            ycol = st.selectbox("Metric", nums, key="line_metric")
            freq = st.selectbox("Frequency", ["D","W","M"], index=2)
            data = df[[dtcol, ycol]].dropna().set_index(dtcol).sort_index().resample(freq).sum()
            fig2 = plt.figure()
            plt.plot(data.index.to_pydatetime(), data[ycol].values.tolist())
            plt.title(f"{ycol} over time ({freq})")
            st.pyplot(fig2)

        if chart == "Histogram" and nums:
            ycol = st.selectbox("Numeric column", nums, key="hist_metric")
            bins = st.slider("Bins", 5, 100, 20)
            series = pd.to_numeric(df[ycol], errors="coerce").dropna()
            fig3 = plt.figure()
            plt.hist(series.values.tolist(), bins=bins)
            plt.title(f"Distribution of {ycol}")
            st.pyplot(fig3)

# ---- EXPORT TAB
with tab_export:
    st.subheader("Export cleaned data & report")
    if st.session_state.cleaned is None:
        st.info("Apply cleaning first.")
    else:
        df = st.session_state.cleaned.copy()
        st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="cleaned.csv", mime="text/csv")
        bufpq = io.BytesIO()
        df.to_parquet(bufpq, index=False)
        st.download_button("Download Parquet", data=bufpq.getvalue(),
                           file_name="cleaned.parquet", mime="application/octet-stream")

        st.markdown("### Export BI-friendly folder")
        if st.button("Prepare BI folder zip"):
            import zipfile, tempfile
            with tempfile.TemporaryDirectory() as td:
                os.makedirs(os.path.join(td, "data"), exist_ok=True)
                df.to_csv(os.path.join(td, "data", "cleaned.csv"), index=False)
                df.to_parquet(os.path.join(td, "data", "cleaned.parquet"), index=False)
                os.makedirs(os.path.join(td, "meta"), exist_ok=True)
                meta = {"columns": df.columns.tolist(), "rows": len(df), "generated_utc": datetime.utcnow().isoformat()}
                with open(os.path.join(td, "meta", "profile.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
                    for root, _, files in os.walk(td):
                        for file in files:
                            full = os.path.join(root, file)
                            rel = os.path.relpath(full, td)
                            z.write(full, rel)
                st.download_button("Download BI folder (zip)", data=zbuf.getvalue(),
                                   file_name="bi_folder.zip", mime="application/zip")

        st.markdown("### Executive summary report")
        lines = []
        lines.append(f"# Executive Summary ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')})")
        lines.append(f"- Rows: {len(df)}  Columns: {len(df.columns)}")
        miss = df.isna().sum().to_frame("missing_count")
        miss["missing_pct"] = (miss["missing_count"] / len(df) * 100).round(2)
        topmiss = miss.sort_values("missing_count", ascending=False).head(10)
        lines.append("## Top Missing Columns")
        for idx, row in topmiss.iterrows():
            lines.append(f"- **{idx}**: {int(row['missing_count'])} ({row['missing_pct']}%)")
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if nums:
            c = nums[0]
            x = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(x) >= 5:
                q1, q3 = x.quantile(0.25), x.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
                out_cnt = int(((x < lo) | (x > hi)).sum())
                lines.append(f"## Outlier check on `{c}`")
                lines.append(f"- IQR bounds ~ [{lo:.2f}, {hi:.2f}]  |  Outliers: {out_cnt}")
        report_md = "\n".join(lines)
        st.download_button("Download Executive Summary (Markdown)", data=report_md.encode("utf-8"),
                           file_name="executive_summary.md", mime="text/markdown")
