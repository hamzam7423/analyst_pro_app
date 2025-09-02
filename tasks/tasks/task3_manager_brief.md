# Task 3: Clean and Analyze Sales Orders

**From:** Sales Manager  
**To:** Data Analyst (You)  
**Date:** Feb 15, 2024  

---

Hi,  

Here’s the raw sales export. Issues include:  

- `Quantity` sometimes contains words instead of numbers (e.g., "five").  
- `Order Date` is in mixed formats (YYYY/MM/DD, DD-MM-YYYY, etc.).  
- Some `Quantity` values are missing (`NaN`).  

---

## What I need from you:
1. Clean the dataset:  
   - Convert all `Quantity` values to numeric (invalid → 0).  
   - Standardize `Order Date` to `YYYY-MM-DD`.  
   - Calculate a new `Total` column = `Quantity * UnitPrice`.  

2. Provide insights:  
   - Total revenue by product.  
   - Total revenue by region.  
   - Monthly sales trend (line chart).  

3. Deliverables:  
   - A clean CSV and Parquet file.  
   - A Markdown summary report.  
   - At least 2 charts (revenue by product + revenue trend over time).  

Thanks,  
*Sales Manager*
