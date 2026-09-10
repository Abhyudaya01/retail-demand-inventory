# Phase 9 Inventory Policy

Phase 9 converts point forecasts into reorder decisions and compares policy cost across the
same 14-day validation demand path used in Phase 7.

## Safety Stock

Safety stock uses `SS = z * forecast_error_std * sqrt(lead_time_days)`. The z-score represents
the desired service level, while the square-root lead-time adjustment assumes daily forecast
errors are stable enough to aggregate over a fixed replenishment lead time.

## Reorder Point

The reorder point is `ROP = mean_daily_demand * lead_time_days + safety_stock`. When inventory
position falls below the reorder point, the policy places an order to restore enough stock for
lead-time demand plus the target days of cycle stock.

## EOQ

Economic order quantity uses `EOQ = sqrt(2 * annual_demand * order_cost / holding_cost)`.
It is included as a textbook reference formula; the backtest uses order-up-to policies because
the project simulates daily store replenishment decisions across a short validation window.

## Default Parameters

`lead_time_days=7` reflects a typical CPG DC-to-store replenishment cycle. `service_level=0.95`
is a common retail planning target. `holding_cost_rate=0.25` approximates annual capital,
warehousing, shrink, and obsolescence cost. `stockout_cost_multiplier=3.0` prices stockouts as
three times unit margin, covering the lost sale plus customer experience and lifetime-value
impact. Opening inventory defaults to two weeks of recent average demand.

## What Would Change In Production

Production planning would use observed lead times by SKU, store and supplier; finance-owned
margin and holding-cost assumptions; order constraints such as case packs and truck schedules;
and category-specific safety-stock policies for perishables, promotion-heavy categories and
intermittent demand.
