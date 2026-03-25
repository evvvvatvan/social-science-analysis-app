# social-science-analysis-app

This is a lightweight Streamlit app that standardizes common quantitative workflows in social science research.

Instead of repeatedly writing ad-hoc code in notebooks, this tool provides a simple interface for:

- Running OLS regressions (with optional fixed effects and robust standard errors)
- Comparing group means with statistical tests and effect sizes
- Exploring correlations with visualizations

The goal is not to replace statistical analysis, but to reduce friction and make routine checks more consistent and reproducible.

## Why I built this

In many research workflows, especially in early-stage analysis, I found myself repeatedly:

- Re-running similar regression models
- Manually computing difference-in-means tables
- Checking correlations across many variables
- Copying and modifying code across notebooks

While these tasks are simple, they are time-consuming and error-prone.

This app is an attempt to turn those repeated steps into a reusable interface — a small “analysis layer” on top of messy data workflows.

## Features

### 1. Regression

- OLS regression with multiple model specifications
- Optional fixed effects (e.g., city-level)
- Optional robust standard errors (HC1)
- Exportable regression tables

### 2. Group Comparison (Difference-in-Means)

- Compare two groups using:
  - Existing categorical variables
  - Threshold rules
  - Top N vs Bottom N splits
- Outputs include:
  - Mean of each group
  - Mean difference
  - p-values (Welch t-test)
  - Significance stars
  - Cohen’s d (effect size)
- Results are sorted by effect size for interpretability

### 3. Correlation Explorer

- Pearson and Spearman correlations
- Scatterplots with optional trendline
- Optional equal-axis comparison (45-degree line)

## Example use cases

- Comparing outcomes between two groups (e.g., treatment vs control, demographic groups)
- Running regression models with consistent specifications across variables
- Identifying meaningful differences using effect sizes (Cohen’s d), not just p-values
- Exploring relationships between variables during early-stage analysis
- Standardizing routine statistical checks to reduce manual errors

## How to run

1. Clone the repository:

```bash
git clone https://github.com/your-username/social-science-analysis-app.git
cd social-science-analysis-app

pip install -r requirements.txt

streamlit run app.py
```

## Notes

- This app is designed for flexibility rather than strict pipelines.
- It assumes users have some familiarity with statistical concepts (OLS, t-tests, etc.).
- It is most useful as a rapid analysis tool, not a final modeling environment.

## Possible extensions

- Clustered standard errors (e.g., by region or tract)
- Percentile-based group comparisons
- Export-ready publication tables
- Integration with geospatial data workflows
