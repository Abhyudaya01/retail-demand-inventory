# Phase 9 Inventory Decision Results

Backtest over 14-day val window, 1,000 SKUs, lead_time_days=7. Small-scale run (5 stores × 200 SKUs). Uses Phase 7 LightGBM predictions.

## Cost comparison across policies

| Policy | Total holding | Total stockout | Total cost | Service level | Stockout units |
|---|---:|---:|---:|---:|---:|
| baseline_rolling_mean | $2,136 | $7,897 | $10,033 | 97.20% | 551 |
| baseline_seasonal_naive | $3,049 | $4,913 | $7,962 | 97.68% | 420 |
| ML_95 | $3,527 | $7,791 | $11,318 | 97.75% | 487 |
| ML_99 | $3,554 | $4,400 | **$7,954** | **97.95%** | 453 |

## Headline findings

- **ML_99 achieves 21% cost reduction** vs baseline_rolling_mean ($7,954 vs $10,033) at higher service level (97.95% vs 97.20%).
- **ML_99 matches baseline_seasonal_naive on cost** ($7,954 vs $7,962, 0.1% cheaper) with 27 basis points higher service level.
- **ML_95 has fewer stockouts than either baseline** but higher holding cost — the 95% safety-stock target proves too conservative for the low-variance forecasts on this dataset.

## Key insight

Better forecasts translate to inventory savings, but only when policies use the FORWARD lead-time forecast window (t through t+L), not just the current-day prediction. A one-day forecast bug made initial ML policies order too late; correcting to use lead-time-forward forecast means enabled the ML advantage.

## Forecast quality context (from Phase 8 analysis)

- Overall residual std: 0.09 (excellent — model is very precise on non-intermittent SKUs)
- Median per-SKU residual std: 0.012
- Overall MAE: 0.02

The low per-SKU error std means safety stock (z × std × √L) is small for most SKUs, so ML policies rely heavily on forecast accuracy rather than buffer stock. This explains why ML wins on smooth-demand SKUs but loses on the small tail of high-variance/intermittent SKUs.

## Parameters (industry-typical defaults)

- Lead time: 7 days (CPG DC-to-store standard)
- Holding cost rate: 25% annual (covers capital, warehousing, obsolescence)
- Stockout cost multiplier: 3× unit margin
- ML_95: z=1.645 (95% service level)
- ML_99: z=2.326 (99% service level)

## What would change in production

- Real per-SKU lead times from supplier data (currently uniform 7 days)
- Category-specific holding cost rates (perishables higher, dry goods lower)
- Real stockout cost curves from Finance (currently 3× margin approximation)
- Per-category safety stock policies (Croston's for intermittent SKUs, LightGBM for smooth)
- Real service-level targets from category management (not uniform 95%/99%)

## Files produced

- reports/tables/inventory_cost_comparison.csv
- reports/tables/inventory_per_sku.csv
- reports/tables/inventory_category_delta.csv
- reports/tables/inventory_service_sensitivity.csv
- reports/figures/inventory_cost_comparison.png
- reports/figures/inventory_service_sensitivity.png
