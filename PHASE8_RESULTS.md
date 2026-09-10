# Phase 8 Error Analysis Results

Segment-level breakdown of LightGBM validation errors from Phase 7 CV (14-day horizon).

## Category-level WAPE (LightGBM)

| Category | Rows | WAPE | Bias |
|---|---|---|---|
| Dairy | 768 | 0.011 | -0.004 |
| Frozen | 488 | 0.010 | 0.000 |
| Garden | 559 | 0.028 | 0.001 |
| Health | 769 | 0.015 | -0.003 |
| Home | 979 | 0.009 | -0.001 |
| Office | 1189 | 0.009 | 0.001 |
| Pet | 1117 | 0.017 | -0.002 |
| Produce | 560 | 0.013 | -0.002 |
| Seasonal | 770 | 0.012 | -0.004 |

**Where the model wins:** Home and Office (WAPE 0.009) — steady low-volume categories with predictable dow patterns.

**Where the model loses:** Garden (WAPE 0.028) — 3× harder than best categories. Seasonal/weather-driven demand not fully captured by current features.

**Bias:** Near-zero across all categories (-0.004 to +0.001) — no systematic over/under-forecasting.

## Intermittency class WAPE

- Smooth SKUs: WAPE 0.055 — model performs well
- Intermittent SKUs: WAPE 1.33 — pushes aggregate WAPE up

**Interpretation:** Aggregate cross-fold WAPE of 0.588 (Phase 7) is inflated by a tail of intermittent-demand SKUs. On smooth-demand SKUs (most SKUs by count), model achieves WAPE 0.055 — an 11× improvement over baseline.

## Worst 5 SKUs (WAPE, filtered to non-intermittent)

| Store | SKU | Category | WAPE |
|---|---|---|---|
| store_0002 | sku_00118 | Garden | 0.093 |
| store_0002 | sku_00144 | Garden | 0.079 |
| store_0004 | sku_00008 | Seasonal | 0.070 |
| store_0004 | sku_00118 | Garden | 0.068 |
| store_0005 | sku_00175 | Office | 0.067 |

Note: 3 of top 5 worst are Garden SKUs — consistent with category-level analysis.

## Best 5 SKUs

| Store | SKU | Category | WAPE |
|---|---|---|---|
| store_0005 | sku_00055 | Toys | 0.002 |
| store_0004 | sku_00015 | Office | 0.002 |
| store_0004 | sku_00172 | Office | 0.002 |
| store_0005 | sku_00038 | Pet | 0.002 |
| store_0005 | sku_00032 | Garden | 0.002 |

## Recommendations for iteration

1. **Add weather features** — Garden's high error suggests weather-driven demand is unmodeled.
2. **Category-specific models** for intermittent-demand SKUs (Croston's method or similar).
3. **Quantile forecasts** for high-variance SKUs where point predictions penalize under both WAPE and business cost.
