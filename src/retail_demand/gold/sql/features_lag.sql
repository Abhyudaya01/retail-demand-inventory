-- Purpose: Demand lag features for autoregressive forecasting.
-- Grain: one row per store_id, sku_id, date.
-- Leakage rationale: LAG only reads rows earlier than date t within each store/SKU series.

SELECT
  store_id,
  sku_id,
  date,
  LAG(units_sold, 1) OVER (
    PARTITION BY store_id, sku_id ORDER BY date
  ) AS lag_1_units,
  LAG(units_sold, 7) OVER (
    PARTITION BY store_id, sku_id ORDER BY date
  ) AS lag_7_units,
  LAG(units_sold, 14) OVER (
    PARTITION BY store_id, sku_id ORDER BY date
  ) AS lag_14_units,
  LAG(units_sold, 28) OVER (
    PARTITION BY store_id, sku_id ORDER BY date
  ) AS lag_28_units,
  LAG(units_sold, 365) OVER (
    PARTITION BY store_id, sku_id ORDER BY date
  ) AS lag_365_units
FROM gold_stg_sales_daily

