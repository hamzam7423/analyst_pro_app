# Task 2: Clean and Analyze Employee Records

**From:** HR Manager  
**To:** Data Analyst (You)  
**Date:** Feb 3, 2024  

---

Hi,  

We exported employee records but the data is messy:  

- The `Full Name` column has inconsistent spacing.  
- `DOB` is sometimes missing or in invalid format (e.g., "not known").  
- Emails have inconsistent casing.  
- Some `Department` values are missing.  
- `Salary` is missing or has `"NaN"`.  

---

## What I need from you:
1. Clean the dataset:  
   - Split `Full Name` into `first_name` and `last_name`.  
   - Convert `DOB` into `YYYY-MM-DD`.  
   - Standardize all emails to lowercase.  
   - Fill missing departments as `"Unknown"`.  
   - Replace missing/NaN salaries with `0`.  

2. Provide insights:  
   - Average salary per department.  
   - Headcount per department.  
   - Identify the department with the highest average salary.  

3. Deliverables:  
   - A clean CSV and Parquet file.  
   - A Markdown summary report.  
   - One chart (bar chart of avg salary per department).  

Thanks,  
*HR Manager*
