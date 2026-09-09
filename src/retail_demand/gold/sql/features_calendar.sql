-- Purpose: Calendar, holiday and cyclical seasonal encodings.
-- Grain: one row per calendar date.
-- Leakage rationale: calendar and holiday dates are known ahead of time and do not use demand outcomes.

WITH holiday_bounds AS (
  SELECT
    c.date,
    c.dow,
    c.week,
    c.month,
    c.quarter,
    c.year,
    c.is_weekend,
    c.is_us_holiday,
    MAX(CASE WHEN c2.is_us_holiday AND c2.date <= c.date THEN c2.date END) AS last_holiday_date,
    MIN(CASE WHEN c2.is_us_holiday AND c2.date >= c.date THEN c2.date END) AS next_holiday_date
  FROM silver_calendar c
  LEFT JOIN silver_calendar c2
    ON c2.date BETWEEN c.date - 370L * 86400000000000L AND c.date + 370L * 86400000000000L
  GROUP BY c.date, c.dow, c.week, c.month, c.quarter, c.year, c.is_weekend, c.is_us_holiday
)
SELECT
  date,
  dow,
  week,
  month,
  quarter,
  year,
  is_weekend,
  is_us_holiday,
  SIN(2.0 * PI() * dow / 7.0) AS dow_sin,
  COS(2.0 * PI() * dow / 7.0) AS dow_cos,
  SIN(2.0 * PI() * month / 12.0) AS month_sin,
  COS(2.0 * PI() * month / 12.0) AS month_cos,
  SIN(2.0 * PI() * DAYOFYEAR(TO_DATE(FROM_UNIXTIME(CAST(date / 1000000000L AS BIGINT)))) / 365.0)
    AS day_of_year_sin,
  COS(2.0 * PI() * DAYOFYEAR(TO_DATE(FROM_UNIXTIME(CAST(date / 1000000000L AS BIGINT)))) / 365.0)
    AS day_of_year_cos,
  CAST((date - last_holiday_date) / 86400000000000L AS INT) AS days_since_last_holiday,
  CAST((next_holiday_date - date) / 86400000000000L AS INT) AS days_to_next_holiday
FROM holiday_bounds

