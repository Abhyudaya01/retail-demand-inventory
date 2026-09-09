-- Purpose: Recent demand trend and intermittency features.
-- Grain: one row per store_id, sku_id, date.
-- Leakage rationale: every rolling window ends at t-1 with ROWS BETWEEN n PRECEDING AND 1 PRECEDING.

SELECT
  store_id,
  sku_id,
  date,
  AVG(units_sold) OVER w7 AS rolling_mean_7,
  STDDEV_SAMP(units_sold) OVER w7 AS rolling_std_7,
  AVG(CASE WHEN units_sold = 0 THEN CAST(1.0 AS DOUBLE) ELSE CAST(0.0 AS DOUBLE) END) OVER w7
    AS rolling_zero_rate_7,
  AVG(units_sold) OVER w14 AS rolling_mean_14,
  STDDEV_SAMP(units_sold) OVER w14 AS rolling_std_14,
  AVG(CASE WHEN units_sold = 0 THEN CAST(1.0 AS DOUBLE) ELSE CAST(0.0 AS DOUBLE) END) OVER w14
    AS rolling_zero_rate_14,
  AVG(units_sold) OVER w28 AS rolling_mean_28,
  STDDEV_SAMP(units_sold) OVER w28 AS rolling_std_28,
  AVG(CASE WHEN units_sold = 0 THEN CAST(1.0 AS DOUBLE) ELSE CAST(0.0 AS DOUBLE) END) OVER w28
    AS rolling_zero_rate_28,
  AVG(units_sold) OVER w91 AS rolling_mean_91,
  STDDEV_SAMP(units_sold) OVER w91 AS rolling_std_91,
  AVG(CASE WHEN units_sold = 0 THEN CAST(1.0 AS DOUBLE) ELSE CAST(0.0 AS DOUBLE) END) OVER w91
    AS rolling_zero_rate_91
FROM gold_stg_sales_daily
WINDOW
  w7 AS (PARTITION BY store_id, sku_id ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING),
  w14 AS (PARTITION BY store_id, sku_id ORDER BY date ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING),
  w28 AS (PARTITION BY store_id, sku_id ORDER BY date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING),
  w91 AS (PARTITION BY store_id, sku_id ORDER BY date ROWS BETWEEN 91 PRECEDING AND 1 PRECEDING)
