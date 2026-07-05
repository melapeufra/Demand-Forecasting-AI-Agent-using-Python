# Demand Forecasting Agent

A lightweight Python agent that forecasts future demand from historical time-series data and recommends production quantities, accounting for safety stock and current inventory levels.

## Features

- **Multiple forecasting methods**: moving average, exponential smoothing, or linear trend
- **Safety stock calculation**: based on a percentage buffer or a statistical service-level formula (whichever is higher)
- **Inventory-aware recommendations**: subtracts current stock from the target to avoid over-producing
- **Multi-group support**: forecasts independently for each group (e.g. Site, UAP, Programme, Famille) in a single run
- **Tidy output**: one row per group per future period, ready to export or plug into a BI tool

## Installation

```bash
pip install pandas numpy --break-system-packages
```

## Usage

```bash
python demand_forecasting_agent.py
```

This runs a demo with synthetic data and writes the results to `demand_forecast_output.csv`.

### Using your own data

Build a DataFrame with these columns:

| Column      | Type              | Description                                      |
|-------------|-------------------|---------------------------------------------------|
| `period`    | datetime          | Historical period (e.g. month start date)          |
| `group`     | string            | Entity to forecast per (Site, UAP, Programme, etc) |
| `demand`    | float             | Historical actual demand / produced quantities    |
| `inventory` | float (optional)  | Current on-hand stock at the time of forecast      |

Then:

```python
from demand_forecasting_agent import DemandForecastingAgent, AgentConfig
import pandas as pd

df = pd.read_csv("my_demand_history.csv", parse_dates=["period"])

config = AgentConfig(
    forecast_horizon=6,        # number of future periods to predict
    method="exp_smoothing",    # "moving_average" | "exp_smoothing" | "linear_trend"
    alpha=0.4,                 # smoothing factor (exp_smoothing only)
    window=3,                  # window size (moving_average only)
    safety_stock_pct=0.15,     # buffer as % of forecasted demand
    service_level_z=1.65,      # ~95% service level
)

agent = DemandForecastingAgent(config)
result = agent.run(df)
result.to_csv("forecast_output.csv", index=False)
```

## Output columns

| Column                        | Description                                      |
|--------------------------------|---------------------------------------------------|
| `group`                        | The forecasted entity                              |
| `period`                       | Future period                                      |
| `forecast_demand`              | Predicted demand for that period                   |
| `safety_stock`                 | Buffer added on top of the forecast                |
| `recommended_production_qty`   | Forecast + safety stock − remaining inventory      |

## Notes

- All three forecasting methods are intentionally simple and dependency-light (no ML libraries required). For higher-accuracy forecasts on seasonal or highly volatile data, consider swapping in a library like `statsmodels` (Holt-Winters / ARIMA) or `prophet`.
- The recommendation logic assumes inventory is consumed period by period against the target; adjust `_recommend_quantities` if your business logic differs (e.g. lead times, batch sizes, minimum order quantities).

## License

MIT
