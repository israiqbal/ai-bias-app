# Quick Error Reference Guide

## Dataset Requirements
- **Minimum Rows:** 20-50 (more is better for accurate results)
- **Target Column:** Must have exactly 2 values with at least 5 samples each
- **Sensitive Attribute:** Must have at least 2 distinct groups with 5+ samples in each
- **Features:** Can be numeric or categorical (auto-encoded)

## Common Error Messages & Solutions

### "❌ Target column must have at least 2 unique values"
**Problem:** Your target column has only 1 value  
**Solution:** 
- Check your column has both positive and negative cases
- Example: target should have values like [approved, denied] not just [approved, approved]
- Verify you selected the correct column

### "❌ Sensitive attribute must have at least 2 unique values"
**Problem:** Your sensitive column has only 1 value  
**Solution:**
- Select a column with 2+ distinct groups (e.g., male/female, not all male)
- Example valid columns: gender, race, age_group, department
- Avoid columns that are all the same value

### "❌ Each class needs at least 5 samples"
**Problem:** One class in your target has fewer than 5 examples  
**Solution:**
- Use a larger dataset
- Example: if "approved: 3, denied: 100" → Need 5+ approved cases
- Balance your dataset or collect more data
- Example fix: Keep rows 1-100 if 5+ of each class

### "❌ Dataset too small after cleanup"
**Problem:** Your dataset has fewer than 20 rows after removing missing values  
**Solution:**
- Use a dataset with at least 20-50 rows
- Check for excessive missing values in your data
- Remove problematic rows before uploading
- Example: "raw data: 18 rows" → Too small, need at least 20

### "❌ No valid features found after encoding"
**Problem:** All your features were removed during encoding  
**Solution:**
- Verify you didn't select the target column as a feature
- Check that you have columns besides target and sensitive attribute
- Example: 
  - ✅ Columns: age, gender, income, target, approved (where target=approved, sensitive=gender)
  - ❌ Columns: only target, sensitive (no other features)
- Keep numeric and categorical features in your dataset

### "❌ Stratified split failed. Using random split."
**Problem:** Not an error - just informational  
**Solution:** No action needed
- This happens with extremely imbalanced datasets
- App automatically falls back to random split
- Analysis will still work correctly

### "⚠️ Removing zero-variance features"
**Problem:** Some features have only one value  
**Solution:** No action needed - removed automatically
- Features with no variation are removed as they don't help the model
- Example: if column "country" is all "USA", it's removed
- This is normal and expected

### ValueError: "This app has encountered an error..."
**Problem:** General model training failure  
**Possible Causes:**
1. Target has only 1 class in training set
2. All features are identical
3. Extreme class imbalance
4. Data types incompatible

**Solutions:**
- Verify both classes have 5+ samples minimum
- Check for columns that are all identical values
- Ensure target column has at least 2 different values
- Try with a different dataset first (test_dataset.csv)

## Quick Checklist Before Uploading

- [ ] CSV file with headers
- [ ] At least 20 rows of data
- [ ] Target column has 2 values with 5+ samples each
- [ ] Sensitive column has 2+ groups with 5+ samples each
- [ ] At least 3-4 feature columns
- [ ] No all-missing columns
- [ ] No all-identical columns (except maybe ID)

## Example Valid Datasets

### ✅ Hiring Data
```
candidate_id, experience_years, education, interview_score, gender, hired
1, 5, Bachelor, 85, Male, yes
2, 3, Masters, 90, Female, yes
... (at least 20 rows, with both yes and no)
```

### ✅ Loan Approval
```
application_id, income, credit_score, age, race, approved
1, 50000, 700, 35, Asian, 1
2, 45000, 650, 28, Black, 0
... (at least 20 rows)
```

### ✅ College Admission
```
student_id, gpa, sat_score, gender, ethnicity, admitted
1, 3.8, 1450, F, White, yes
2, 3.5, 1350, M, Hispanic, no
... (at least 20 rows)
```

### ❌ Invalid Examples

**Too Few Rows**
```
(only 5 rows) → Need at least 20
```

**Single Class Only**
```
target column: [yes, yes, yes, yes...] → Need yes AND no
```

**Single Group Only**
```
gender column: [M, M, M, M...] → Need M AND F (or equivalent)
```

**No Features**
```
Columns: just "target", "gender" → Need additional features
```

## Testing Your Data

Before uploading to the app, verify in Python:

```python
import pandas as pd

df = pd.read_csv('your_data.csv')

# Check size
print(f"Rows: {len(df)}")  # Should be >= 20

# Check target
target = 'approved'  # Replace with your column name
print(f"Target values: {df[target].value_counts()}")
# Should show at least 2 values with 5+ each

# Check sensitive attribute
sensitive = 'gender'  # Replace with your column name
print(f"Sensitive values: {df[sensitive].value_counts()}")
# Should show at least 2 groups with 5+ each

# Check for missing
print(f"Missing: {df.isnull().sum()}")
# Should be minimal
```

## Need More Help?

1. Check the "How to Use This Tool" section in the app
2. Review the "Troubleshooting" section in the app
3. Compare your data with example datasets provided
4. Verify your dataset meets the Quick Checklist above

---
Last Updated: 2026-05-18
