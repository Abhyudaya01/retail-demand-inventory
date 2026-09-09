-- Purpose: Prior-year seasonal demand anchors.
-- Grain: one row per store_id, sku_id, date.
-- Leakage rationale: same-year comparisons use LAG and windows ending at least 365 days before t.

SELECT
  store_id,
  sku_id,
  date,
  LAG(units_sold, 365) OVER (
    PARTITION BY store_id, sku_id ORDER BY date
  ) AS same_day_last_year_units,
  AVG(units_sold) OVER (
    PARTITION BY store_id, sku_id ORDER BY date ROWS BETWEEN 371 PRECEDING AND 365 PRECEDING
  ) AS same_week_last_year_mean,
  CASE
    WHEN AVG(units_sold) OVER (
      PARTITION BY store_id, sku_id ORDER BY date ROWS BETWEEN 371 PRECEDING AND 365 PRECEDING
    ) = 0 THEN NULL
    ELSE LAG(units_sold, 1) OVER (
      PARTITION BY store_id, sku_id ORDER BY date
    ) / AVG(units_sold) OVER (
      PARTITION BY store_id, sku_id ORDER BY date ROWS BETWEEN 371 PRECEDING AND 365 PRECEDING
    )
  END AS yoy_growth_ratio
FROM gold_stg_sales_daily

