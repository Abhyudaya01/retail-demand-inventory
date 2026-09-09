-- Purpose: Time-aware price and discount features.
-- Grain: one row per store_id, sku_id, date.
-- Leakage rationale: each sale date joins only to the latest price week_start where week_start <= date.

WITH price_asof AS (
  SELECT
    s.store_id,
    s.sku_id,
    s.date,
    p.week_start,
    p.price,
    p.is_promo,
    ROW_NUMBER() OVER (
      PARTITION BY s.store_id, s.sku_id, s.date
      ORDER BY p.week_start DESC
    ) AS rn
  FROM gold_stg_sales_daily s
  LEFT JOIN silver_prices p
    ON s.store_id = p.store_id
   AND s.sku_id = p.sku_id
   AND p.week_start <= s.date
),
current_price AS (
  SELECT
    store_id,
    sku_id,
    date,
    week_start,
    price AS current_price,
    CAST(is_promo AS BOOLEAN) AS is_on_promo
  FROM price_asof
  WHERE rn = 1
),
with_prior AS (
  SELECT
    c.*,
    p4.price AS price_4w_ago
  FROM current_price c
  LEFT JOIN silver_prices p4
    ON c.store_id = p4.store_id
   AND c.sku_id = p4.sku_id
   AND p4.week_start = c.week_start - 28L * 24L * 60L * 60L * 1000000000L
)
SELECT
  store_id,
  sku_id,
  date,
  week_start AS price_week_start,
  current_price,
  price_4w_ago,
  current_price / NULLIF(price_4w_ago, 0.0) AS price_ratio_4w,
  CASE
    WHEN price_4w_ago IS NULL OR price_4w_ago = 0.0 THEN NULL
    ELSE GREATEST(0.0, (price_4w_ago - current_price) / price_4w_ago)
  END AS discount_depth,
  is_on_promo
FROM with_prior

