-- Purpose: ML-ready feature matrix for demand forecasting.
-- Grain: one row per store_id, sku_id, date with label units_sold at t.
-- Leakage rationale: label is separated from features; joined feature families use lagged demand,
-- as-of prices, calendar facts known ahead of time, and prior-year windows that do not read after t.

SELECT
  s.store_id,
  s.sku_id,
  s.date,
  s.units_sold AS label_units_sold,
  s.revenue,
  l.lag_1_units,
  l.lag_7_units,
  l.lag_14_units,
  l.lag_28_units,
  l.lag_365_units,
  r.rolling_mean_7,
  r.rolling_std_7,
  r.rolling_zero_rate_7,
  r.rolling_mean_14,
  r.rolling_std_14,
  r.rolling_zero_rate_14,
  r.rolling_mean_28,
  r.rolling_std_28,
  r.rolling_zero_rate_28,
  r.rolling_mean_91,
  r.rolling_std_91,
  r.rolling_zero_rate_91,
  p.price_week_start,
  p.current_price,
  p.price_4w_ago,
  p.price_ratio_4w,
  p.discount_depth,
  p.is_on_promo,
  pr.days_since_promo_start,
  pr.days_until_promo_end,
  pr.promo_streak_length,
  c.dow,
  c.week,
  c.month,
  c.quarter,
  c.year,
  c.is_weekend,
  c.is_us_holiday,
  c.dow_sin,
  c.dow_cos,
  c.month_sin,
  c.month_cos,
  c.day_of_year_sin,
  c.day_of_year_cos,
  c.days_since_last_holiday,
  c.days_to_next_holiday,
  se.same_day_last_year_units,
  se.same_week_last_year_mean,
  se.yoy_growth_ratio
FROM gold_stg_sales_daily s
LEFT JOIN gold_features_lag l
  ON s.store_id = l.store_id AND s.sku_id = l.sku_id AND s.date = l.date
LEFT JOIN gold_features_rolling r
  ON s.store_id = r.store_id AND s.sku_id = r.sku_id AND s.date = r.date
LEFT JOIN gold_features_price p
  ON s.store_id = p.store_id AND s.sku_id = p.sku_id AND s.date = p.date
LEFT JOIN gold_features_promo pr
  ON s.store_id = pr.store_id AND s.sku_id = pr.sku_id AND s.date = pr.date
LEFT JOIN gold_features_calendar c
  ON s.date = c.date
LEFT JOIN gold_features_seasonal se
  ON s.store_id = se.store_id AND s.sku_id = se.sku_id AND s.date = se.date

