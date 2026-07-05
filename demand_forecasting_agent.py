"""
Demand Forecasting Agent
=========================
Predicts future demand from historical time-series data and recommends
production quantities per period, accounting for safety stock and
current inventory levels.

Usage:
    python demand_forecasting_agent.py

Dependencies:
    pip install pandas numpy --break-system-packages
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    forecast_horizon: int = 6          # number of future periods to forecast
    method: str = "exp_smoothing"      # "moving_average" | "exp_smoothing" | "linear_trend"
    window: int = 3                    # window size for moving average
    alpha: float = 0.4                 # smoothing factor for exponential smoothing
    safety_stock_pct: float = 0.15     # buffer as % of forecasted demand
    service_level_z: float = 1.65      # z-score for ~95% service level (used with demand std)
    min_production_qty: float = 0.0    # floor for recommended production


# ---------------------------------------------------------------------------
# Core forecasting engine
# ---------------------------------------------------------------------------

class DemandForecastingAgent:
    """
    Wraps a forecasting method + a production recommendation rule.

    Input data expected as a long-format DataFrame with columns:
        - "period" (datetime or sortable period label)
        - "group" (e.g. Site, UAP, Programme, Famille - the entity to forecast per)
        - "demand" (historical actual demand / produced quantities)

    Optionally:
        - "inventory" (current on-hand stock at time of forecast, per group)
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()

    # -- Forecast methods ----------------------------------------------------

    def _moving_average(self, series: np.ndarray) -> np.ndarray:
        w = self.config.window
        last_window = series[-w:] if len(series) >= w else series
        level = last_window.mean()
        return np.full(self.config.forecast_horizon, level)

    def _exp_smoothing(self, series: np.ndarray) -> np.ndarray:
        alpha = self.config.alpha
        level = series[0]
        for value in series[1:]:
            level = alpha * value + (1 - alpha) * level
        return np.full(self.config.forecast_horizon, level)

    def _linear_trend(self, series: np.ndarray) -> np.ndarray:
        x = np.arange(len(series))
        if len(series) < 2:
            return np.full(self.config.forecast_horizon, series[-1])
        slope, intercept = np.polyfit(x, series, 1)
        future_x = np.arange(len(series), len(series) + self.config.forecast_horizon)
        forecast = slope * future_x + intercept
        return np.clip(forecast, a_min=0, a_max=None)

    def _forecast_series(self, series: np.ndarray) -> np.ndarray:
        method_map = {
            "moving_average": self._moving_average,
            "exp_smoothing": self._exp_smoothing,
            "linear_trend": self._linear_trend,
        }
        if self.config.method not in method_map:
            raise ValueError(f"Unknown method: {self.config.method}")
        return method_map[self.config.method](series)

    # -- Production recommendation -------------------------------------------

    def _recommend_quantities(
        self, forecast: np.ndarray, demand_std: float, inventory: float = 0.0
    ) -> np.ndarray:
        cfg = self.config
        safety_stock = np.maximum(
            forecast * cfg.safety_stock_pct,
            cfg.service_level_z * demand_std,
        )
        target = forecast + safety_stock

        recommended = np.empty_like(target)
        remaining_inventory = inventory
        for i, t in enumerate(target):
            qty = max(t - remaining_inventory, cfg.min_production_qty)
            recommended[i] = qty
            # any inventory beyond this period's target carries forward as 0
            # (produced qty consumed to meet target, no leftover assumed)
            remaining_inventory = max(remaining_inventory - t, 0.0)
        return recommended

    # -- Public API ------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs forecasting + recommendation for every group in df.
        Returns a tidy DataFrame with one row per (group, future_period).
        """
        required_cols = {"period", "group", "demand"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Input dataframe missing columns: {missing}")

        results = []
        for group_name, group_df in df.sort_values("period").groupby("group"):
            series = group_df["demand"].to_numpy(dtype=float)
            demand_std = series.std(ddof=0) if len(series) > 1 else 0.0
            inventory = (
                group_df["inventory"].iloc[-1] if "inventory" in group_df.columns else 0.0
            )

            forecast = self._forecast_series(series)
            recommended = self._recommend_quantities(forecast, demand_std, inventory)

            last_period = group_df["period"].max()
            future_periods = self._build_future_periods(last_period, self.config.forecast_horizon)

            for period, fcst, rec in zip(future_periods, forecast, recommended):
                results.append(
                    {
                        "group": group_name,
                        "period": period,
                        "forecast_demand": round(float(fcst), 1),
                        "safety_stock": round(float(rec - fcst), 1) if rec >= fcst else 0.0,
                        "recommended_production_qty": round(float(rec), 1),
                    }
                )

        return pd.DataFrame(results)

    @staticmethod
    def _build_future_periods(last_period, horizon: int) -> list:
        if isinstance(last_period, (pd.Timestamp,)) or np.issubdtype(type(last_period), np.datetime64):
            last_period = pd.Timestamp(last_period)
            return [last_period + pd.DateOffset(months=i + 1) for i in range(horizon)]
        # fallback: integer/label periods -> just increment
        return [f"{last_period}+{i + 1}" for i in range(horizon)]


# ---------------------------------------------------------------------------
# Demo / self-test with synthetic data
# ---------------------------------------------------------------------------

def _generate_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    periods = pd.date_range("2025-01-01", periods=12, freq="MS")
    groups = ["Famille", "Site", "UAP"]

    rows = []
    for group in groups:
        base = rng.uniform(50, 150)
        trend = rng.uniform(-1.5, 2.5)
        for i, period in enumerate(periods):
            noise = rng.normal(0, base * 0.08)
            demand = max(base + trend * i + noise, 0)
            rows.append(
                {
                    "period": period,
                    "group": group,
                    "demand": round(demand, 1),
                    "inventory": round(rng.uniform(0, base * 0.3), 1) if i == len(periods) - 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def main():
    print("Génération de données de démonstration...\n")
    df = _generate_demo_data()

    config = AgentConfig(
        forecast_horizon=6,
        method="exp_smoothing",
        alpha=0.4,
        safety_stock_pct=0.15,
        service_level_z=1.65,
    )
    agent = DemandForecastingAgent(config)
    output = agent.run(df)

    output["period"] = output["period"].apply(
        lambda p: p.strftime("%m-%Y") if isinstance(p, pd.Timestamp) else p
    )

    print("Prévisions de demande et recommandations de production :\n")
    print(output.to_string(index=False))

    output.to_csv("/mnt/user-data/outputs/demand_forecast_output.csv", index=False)
    print("\nRésultats sauvegardés dans demand_forecast_output.csv")


if __name__ == "__main__":
    main()
