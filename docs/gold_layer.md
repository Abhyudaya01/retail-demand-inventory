# Gold layer

Gold builds analytics-ready forecasting features from path-backed Silver Delta tables. It
does not register Unity Catalog tables; Databricks Free Edition reads Silver from
`/Volumes/workspace/retail_demand/silver` and writes Gold to
`/Volumes/workspace/retail_demand/gold`.

Feature engineering lives in `src/retail_demand/gold/sql/*.sql`. The builder creates temp
views from Delta paths, runs each SQL file, adds `_gold_run_id`, and writes the result as
Delta partitioned by `year` and `month` when a `date` column is present.

## Feature Dictionary

| Feature | Formula | Purpose | Leakage guard |
| --- | --- | --- | --- |
| `lag_1_units` | `LAG(units_sold, 1)` | Yesterday demand | Reads only t-1 |
| `lag_7_units` | `LAG(units_sold, 7)` | Weekly seasonality | Reads only t-7 |
| `lag_14_units` | `LAG(units_sold, 14)` | Two-week recurrence | Reads only t-14 |
| `lag_28_units` | `LAG(units_sold, 28)` | Monthly recurrence | Reads only t-28 |
| `lag_365_units` | `LAG(units_sold, 365)` | Annual recurrence | Reads only t-365 |
| `rolling_mean_{7,14,28,91}` | mean over rows ending t-1 | Recent demand level | Window excludes current row |
| `rolling_std_{7,14,28,91}` | sample stddev ending t-1 | Demand volatility | Window excludes current row |
| `rolling_zero_rate_{7,14,28,91}` | zero-unit share ending t-1 | Intermittency | Window excludes current row |
| `current_price` | latest price week where `week_start <= date` | As-of price | No future price weeks |
| `price_4w_ago` | price at current price week minus 28 days | Price anchor | Historical lookup |
| `price_ratio_4w` | current price / price 4 weeks ago | Price movement | Uses as-of and past prices |
| `discount_depth` | positive discount from 4-week anchor | Promo intensity | Uses as-of and past prices |
| `is_on_promo` | as-of `is_promo` | Promo flag | No future price weeks |
| `days_since_promo_start` | days since current promo block start | Promo age | Current known block |
| `days_until_promo_end` | days until current promo block end | Promo horizon | Calendarized price plan |
| `promo_streak_length` | current promo-block day count | Promo duration | Current known block |
| `dow_sin`, `dow_cos` | cyclic encoding over 7 days | Weekly pattern | Calendar-known |
| `month_sin`, `month_cos` | cyclic encoding over 12 months | Annual pattern | Calendar-known |
| `day_of_year_sin`, `day_of_year_cos` | cyclic encoding over 365 days | Seasonal position | Calendar-known |
| `is_us_holiday` | Silver calendar flag | Holiday effect | Calendar-known |
| `days_to_next_holiday` | next holiday distance | Planning signal | Calendar-known |
| `days_since_last_holiday` | last holiday distance | Holiday aftermath | Calendar-known |
| `same_day_last_year_units` | `LAG(units_sold, 365)` | Annual demand memory | Reads only t-365 |
| `same_week_last_year_mean` | mean over t-371 to t-365 | Annual weekly anchor | Ends at least 365 days before t |
| `yoy_growth_ratio` | yesterday units / same-week-last-year mean | Trend scale | Uses t-1 and prior-year window |

## SQL Excerpts

Lag features:

```sql
LAG(units_sold, 7) OVER (
  PARTITION BY store_id, sku_id ORDER BY date
) AS lag_7_units
```

Rolling features:

```sql
AVG(units_sold) OVER (
  PARTITION BY store_id, sku_id ORDER BY date
  ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
) AS rolling_mean_7
```

Price features:

```sql
LEFT JOIN silver_prices p
  ON s.store_id = p.store_id
 AND s.sku_id = p.sku_id
 AND p.week_start <= s.date
```

Calendar features:

```sql
SIN(2.0 * PI() * dow / 7.0) AS dow_sin
```

Seasonal features:

```sql
AVG(units_sold) OVER (
  PARTITION BY store_id, sku_id ORDER BY date
  ROWS BETWEEN 371 PRECEDING AND 365 PRECEDING
) AS same_week_last_year_mean
```

## Leakage Audit

`audit_features_master(spark, gold_root)` samples up to 1000 rows from `features_master` and
recomputes expected lag, rolling, and price features from the path-backed Gold tables. It
prints offending rows and raises `ValueError` when a mismatch is found. A clean audit returns
one result per checked feature with `passes`, `sample_size_checked`, and `mismatches`.

## Local Execution

Build Bronze and Silver first, then run Gold:

```bash
make bronze-local
make silver-local
make gold-local
```

For custom paths:

```bash
python -m retail_demand.gold.cli --silver-root data/silver --gold-root data/gold
```

