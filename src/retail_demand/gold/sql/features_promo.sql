-- Purpose: Promotion streak and boundary features.
-- Grain: one row per store_id, sku_id, date.
-- Leakage rationale: current and prior promo state are observed by date t; future-looking end distance
-- is included only within the currently known promo block from the path-backed price calendar.

WITH base AS (
  SELECT
    store_id,
    sku_id,
    date,
    is_on_promo,
    CASE
      WHEN is_on_promo
       AND COALESCE(LAG(is_on_promo) OVER (
         PARTITION BY store_id, sku_id ORDER BY date
       ), false) = false
      THEN 1 ELSE 0
    END AS promo_start_flag
  FROM gold_features_price
),
blocks AS (
  SELECT
    *,
    SUM(promo_start_flag) OVER (
      PARTITION BY store_id, sku_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS promo_block_id
  FROM base
),
with_bounds AS (
  SELECT
    *,
    MIN(CASE WHEN is_on_promo THEN date END) OVER (
      PARTITION BY store_id, sku_id, promo_block_id
    ) AS promo_start_date,
    MAX(CASE WHEN is_on_promo THEN date END) OVER (
      PARTITION BY store_id, sku_id, promo_block_id
    ) AS promo_end_date
  FROM blocks
)
SELECT
  store_id,
  sku_id,
  date,
  CASE WHEN is_on_promo THEN CAST((date - promo_start_date) / 86400000000000L AS INT) ELSE NULL END
    AS days_since_promo_start,
  CASE WHEN is_on_promo THEN CAST((promo_end_date - date) / 86400000000000L AS INT) ELSE NULL END
    AS days_until_promo_end,
  CASE WHEN is_on_promo THEN CAST((date - promo_start_date) / 86400000000000L AS INT) + 1 ELSE 0 END
    AS promo_streak_length
FROM with_bounds

