"""Daily/weekly/monthly aggregations and rule-based insights."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd


def _day_key(created_at: str) -> str:
    """Expect ISO local datetime string; return YYYY-MM-DD."""
    if not created_at:
        return ""
    return created_at[:10]


def aggregate_by_day(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Sum macros per calendar day (local date prefix)."""
    out: dict[str, dict[str, float]] = defaultdict(
        lambda: {"Calories": 0.0, "Protein": 0.0, "Carbs": 0.0, "Fats": 0.0}
    )
    for r in rows:
        dk = _day_key(str(r.get("created_at", "")))
        if not dk:
            continue
        out[dk]["Calories"] += float(r.get("Calories", 0) or 0)
        out[dk]["Protein"] += float(r.get("Protein", 0) or 0)
        out[dk]["Carbs"] += float(r.get("Carbs", 0) or 0)
        out[dk]["Fats"] += float(r.get("Fats", 0) or 0)
    return dict(out)


def daily_totals_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    by_day = aggregate_by_day(rows)
    if not by_day:
        return pd.DataFrame(columns=["date", "Calories", "Protein", "Carbs", "Fats"])
    records = []
    for dk in sorted(by_day.keys()):
        m = by_day[dk]
        records.append(
            {
                "date": dk,
                "Calories": m["Calories"],
                "Protein": m["Protein"],
                "Carbs": m["Carbs"],
                "Fats": m["Fats"],
            }
        )
    return pd.DataFrame(records)


def period_averages(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean daily totals over days that have any logged food."""
    df = daily_totals_dataframe(rows)
    if df.empty:
        return {"Calories": 0, "Protein": 0, "Carbs": 0, "Fats": 0}
    return {
        "Calories": float(df["Calories"].mean()),
        "Protein": float(df["Protein"].mean()),
        "Carbs": float(df["Carbs"].mean()),
        "Fats": float(df["Fats"].mean()),
    }


def rule_based_insights(
    rows_last_7d: list[dict[str, Any]],
    goals: dict[str, float | int],
) -> list[str]:
    """Explainable rules over the last 7 days of data."""
    insights: list[str] = []
    if not rows_last_7d:
        return ["Log a few meals to unlock trend insights."]

    by_day = aggregate_by_day(rows_last_7d)
    if not by_day:
        return ["Log a few meals to unlock trend insights."]

    days = sorted(by_day.keys())
    n_days = len(days)
    p_goal = float(goals.get("Protein", 150))
    c_goal = float(goals.get("Calories", 2000))

    low_protein_days = sum(
        1 for d in days if by_day[d]["Protein"] < p_goal * 0.85
    )
    if low_protein_days >= 3:
        insights.append(
            f"Protein was below ~85% of your goal on {low_protein_days} of the last {n_days} logging day(s). "
            "Consider adding a lean protein source to one meal per day."
        )

    high_cal_days = sum(1 for d in days if by_day[d]["Calories"] > c_goal * 1.1)
    if high_cal_days >= 4:
        insights.append(
            f"Calories exceeded ~110% of target on {high_cal_days} day(s) in this window. "
            "Check portions or liquid calories if that was unintentional."
        )

    low_cal_days = sum(1 for d in days if 0 < by_day[d]["Calories"] < c_goal * 0.7)
    if low_cal_days >= 4:
        insights.append(
            f"Several days were well under your calorie goal ({low_cal_days} day(s)). "
            "If you are not aiming for a deficit, add nutrient-dense snacks."
        )

    if not insights:
        insights.append(
            "Your intake in this window is relatively balanced vs your goals. Keep logging for stronger trends."
        )

    return insights


def suggestion_cards(
    rows_today: list[dict[str, Any]],
    goals: dict[str, float | int],
) -> list[str]:
    """Actionable nudges based on today's totals vs goals."""
    sug: list[str] = []
    by_day = aggregate_by_day(rows_today)
    today_str = date.today().isoformat()
    today = by_day.get(today_str, {"Calories": 0, "Protein": 0, "Carbs": 0, "Fats": 0})

    p_goal = float(goals.get("Protein", 150))
    c_goal = float(goals.get("Calories", 2000))
    gap_p = p_goal - today["Protein"]
    if gap_p > 15:
        sug.append(
            f"About {gap_p:.0f} g protein left to hit today's target — try Greek yogurt, eggs, tofu, or lentils."
        )

    gap_c = c_goal - today["Calories"]
    if gap_c > 200:
        sug.append(
            f"Roughly {gap_c:.0f} kcal remaining today — add a balanced snack (fruit + nuts, or whole-grain + protein)."
        )
    elif today["Calories"] > c_goal * 1.15:
        over = today["Calories"] - c_goal
        sug.append(
            f"You're about {over:.0f} kcal over today's goal — lighter dinner or a walk can help balance the week."
        )

    if not sug:
        sug.append("Goals look on track for today. Log your next meal to keep insights accurate.")

    return sug


def date_range_last_n_days(n: int) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=n - 1)
    return start, end
