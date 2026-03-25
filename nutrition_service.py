"""Hybrid nutrition: Open Food Facts lookup first, LLM estimate as fallback."""

from __future__ import annotations

import json
import re
from typing import Any

import requests

import utils

OFF_SEARCH = "https://world.openfoodfacts.org/cgi/search.pl"
REQUEST_TIMEOUT = 12

# Heuristic grams per "whole" unit when product has only per-100g data
DEFAULT_GRAMS_PER_UNIT: dict[str, float] = {
    "egg": 50,
    "eggs": 50,
    "apple": 180,
    "banana": 120,
    "slice": 30,
    "piece": 100,
    "whole": 100,
    "serving": 150,
    "cup": 240,
    "bowl": 250,
    "tbsp": 15,
    "tsp": 5,
}


def _normalize_unit(unit: str | None) -> str:
    if not unit:
        return "serving"
    u = unit.strip().lower()
    if u in ("g", "gram", "grams"):
        return "g"
    if u in ("ml", "milliliter", "milliliters"):
        return "ml"
    return u


def _grams_for_serving(quantity: float, unit: str, food_name: str) -> float | None:
    u = _normalize_unit(unit)
    if u == "g":
        return max(quantity, 0)
    if u == "ml":
        return max(quantity, 0)
    if u in DEFAULT_GRAMS_PER_UNIT:
        return max(quantity, 0) * DEFAULT_GRAMS_PER_UNIT[u]
    # guess from food name keywords
    name_l = (food_name or "").lower()
    for key, g in DEFAULT_GRAMS_PER_UNIT.items():
        if key in name_l and key not in ("whole", "piece", "serving"):
            return max(quantity, 0) * g
    return max(quantity, 0) * DEFAULT_GRAMS_PER_UNIT["serving"]


def _pick_nutriments(product: dict[str, Any]) -> dict[str, float] | None:
    nut = product.get("nutriments") or {}
    kcal = nut.get("energy-kcal_100g")
    if kcal is None:
        kcal = nut.get("energy-kcal")
    if kcal is None and nut.get("energy-kj_100g"):
        kcal = float(nut["energy-kj_100g"]) / 4.184
    if kcal is None:
        return None
    p = nut.get("proteins_100g") or 0
    c = nut.get("carbohydrates_100g") or 0
    f = nut.get("fat_100g") or 0
    try:
        return {
            "kcal": float(kcal),
            "protein": float(p),
            "carbs": float(c),
            "fats": float(f),
        }
    except (TypeError, ValueError):
        return None


def lookup_open_food_facts(
    food_name: str, quantity: float | None, unit: str | None
) -> tuple[dict[str, float], str] | None:
    """
    Returns ({Calories, Protein, Carbs, Fats}, product_display_name) or None.
    """
    q = re.sub(r"\s+", " ", (food_name or "").strip())[:80]
    if len(q) < 2:
        return None
    try:
        r = requests.get(
            OFF_SEARCH,
            params={
                "search_terms": q,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": 5,
            },
            headers={"User-Agent": "NutriVoice/1.0 (Streamlit local app)"},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError):
        return None

    products = data.get("products") or []
    best = None
    for prod in products:
        per = _pick_nutriments(prod)
        if per and per["kcal"] > 0:
            best = (prod, per)
            break
    if not best:
        return None
    prod, per = best
    pname = (prod.get("product_name") or prod.get("product_name_en") or food_name).strip()
    qty = float(quantity) if quantity is not None else 1.0
    grams = _grams_for_serving(qty, unit or "serving", food_name)
    if grams is None or grams <= 0:
        return None
    factor = grams / 100.0
    return (
        {
            "Calories": round(per["kcal"] * factor),
            "Protein": round(per["protein"] * factor, 1),
            "Carbs": round(per["carbs"] * factor, 1),
            "Fats": round(per["fats"] * factor, 1),
        },
        pname,
    )


def resolve_meal_text(
    text: str, api_key: str
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Extract entities, then OFF lookup per item, then batched LLM fallback for misses.
    Returns (items, meal_type) suitable for data_store / UI.
    """
    if not text.strip():
        return [], None
    entities = utils.extract_food_entities(text, api_key)
    if not entities:
        return [], None
    meal_type = entities[0].get("meal_type")
    if isinstance(meal_type, str):
        meal_type = meal_type.strip().lower() or None
    else:
        meal_type = None

    resolved: list[dict[str, Any]] = []
    need_fallback: list[dict[str, Any]] = []

    for ent in entities:
        name = ent.get("name") or "Unknown"
        qty = ent.get("quantity")
        unit = ent.get("unit")
        try:
            qf = float(qty) if qty is not None else None
        except (TypeError, ValueError):
            qf = None

        off = lookup_open_food_facts(name, qf, unit)
        display = name
        if off:
            macros, pname = off
            display = pname or name
            # Low score if search was very generic
            conf = "matched"
            if len((name or "").split()) <= 1 and len(pname or "") > 40:
                conf = "needs_review"
            resolved.append(
                {
                    "Name": display,
                    "quantity": qf,
                    "unit": unit,
                    "Calories": int(macros["Calories"]),
                    "Protein": float(macros["Protein"]),
                    "Carbs": float(macros["Carbs"]),
                    "Fats": float(macros["Fats"]),
                    "nutrition_source": "db",
                    "confidence_status": conf,
                }
            )
        else:
            need_fallback.append(
                {
                    "name": name,
                    "quantity": qf,
                    "unit": unit,
                    "meal_type": ent.get("meal_type"),
                }
            )

    if need_fallback:
        estimates = utils.estimate_nutrition_batch(need_fallback, api_key)
        for est in estimates:
            resolved.append(
                {
                    "Name": est.get("Name", "Unknown"),
                    "quantity": est.get("quantity"),
                    "unit": est.get("unit"),
                    "Calories": int(est.get("Calories", 0)),
                    "Protein": float(est.get("Protein", 0)),
                    "Carbs": float(est.get("Carbs", 0)),
                    "Fats": float(est.get("Fats", 0)),
                    "nutrition_source": "llm_fallback",
                    "confidence_status": "fallback",
                }
            )

    return resolved, meal_type


def transcribe_audio_only(audio_bytes: bytes, mime_type: str) -> tuple[str | None, str | None]:
    """
    Returns (transcript, error_message). No LLM.
    """
    return utils.transcribe_audio_bytes(audio_bytes, mime_type)
