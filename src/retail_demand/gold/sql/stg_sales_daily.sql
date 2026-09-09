-- Purpose: Canonical daily sales grain used by all Gold feature families.
-- Grain: one row per store_id, sku_id, date from Silver sales.
-- Leakage rationale: contains only observed label columns at date t and no future joins.

SELECT
  store_id,
  sku_id,
  date,
  units_sold,
  revenue,
  year,
  month
FROM silver_sales

