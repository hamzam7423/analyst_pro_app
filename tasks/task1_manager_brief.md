# Task 1: Clean and Analyze Company Earnings

**From:** Manager  
**To:** Data Analyst (You)  
**Date:** Jan 10, 2024  

---

Hi,  

I’ve attached the latest *Company Earnings* export from our finance system. Unfortunately, the data is messy again:  

- Company names are inconsistent (sometimes lowercase, sometimes uppercase).  
- The `Date` column is in mixed formats.  
- Some `Revenue($)` values are missing or marked as `"n/a"`.  
- We need to check for **duplicate invoices**.  

---

## What I need from you:
1. Clean the dataset so it’s ready for analysis:  
   - Standardize company names (Title Case).  
   - Parse all dates into `YYYY-MM-DD`.  
   - Replace missing or `"n/a"` revenues with `0`.  
   - Deduplicate by `Invoice_ID`.  

2. Provide insights:  
   - Total revenue per company.  
   - Monthly revenue trend.  
   - Identify top 3 companies by revenue.  

3. Deliverables:  
   - A **clean CSV** and **Parquet file**.  
   - A **short Markdown summary report** (rows, columns, missing values, and top 3 companies).  
   - One or two simple **charts** (bar or line).  

Thanks,  
*Your Manager*
