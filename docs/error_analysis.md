# Phase 8 Error Analysis

Phase 8 interprets the best Phase 7 LightGBM predictions by segment. It is meant to answer
where the model wins, where it loses, and which errors are systematic enough to guide the next
modeling or merchandising iteration.

## Where The Model Wins

High-volume smooth SKUs on non-promo days are expected to be the strongest segment. Fill this
section with the lowest-WAPE volume and intermittency segments after running
`notebooks/analysis/06_error_analysis.ipynb` against the full LightGBM prediction artifact.

## Where The Model Loses

Intermittent/lumpy SKUs, promo days, and [category to fill] are expected weak spots. Replace the
bracketed category with the highest-WAPE category from `reports/tables/`.

## Systemic Biases

[categories from bias_flags] should be listed here with over/under-forecast direction and bias
ratio after the notebook writes the bias flag tables.

## Recommendations For Iteration

Specific next steps: add richer promo timing and depth features, evaluate category-specific
models, test quantile regression or Croston-style features for intermittent demand, and review
high-bias store/category intersections with merchandising context.

## Reproducibility

Default MLflow run id: `28cb96fe2b5542c797b048dffbb6eae6`. Dataset scale: Phase 7 Databricks
LightGBM predictions from the path-backed Gold `features_master` table, with the exact row count
recorded in the executed notebook output.
