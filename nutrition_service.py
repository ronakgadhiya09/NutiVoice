"""Robust nutrition pipeline: normalize -> DB match -> LLM fallback -> validate/correct."""

from __future__ import annotations

import difflib
import json
import re
from functools import lru_cache
from typing import Any

import requests

import utils

OFF_SEARCH = "https://world.openfoodfacts.org/cgi/search.pl"
REQUEST_TIMEOUT = 12

CONFIDENCE_ORDER = ["low", "medium-low", "medium", "high"]

BRAND_NOISE_TOKENS = {
    "minute",
    "meals",
    "double",
    "extra",
    "brand",
    "homestyle",
    "signature",
    "ready",
    "instant",
    "frozen",
    "fresh",
    "classic",
    "premium",
    "style",
}

UNIT_ALIASES: dict[str, str] = {
    "gram": "g",
    "grams": "g",
    "gm": "g",
    "g": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "kg": "kg",
    "milliliter": "ml",
    "milliliters": "ml",
    "ml": "ml",
    "litre": "l",
    "liter": "l",
    "liters": "l",
    "l": "l",
    "egg": "eggs",
    "eggs": "eggs",
    "piece": "piece",
    "pieces": "piece",
    "slice": "slice",
    "slices": "slice",
    "cup": "cup",
    "cups": "cup",
    "bowl": "bowl",
    "bowls": "bowl",
    "plate": "plate",
    "plates": "plate",
    "serving": "serving",
    "servings": "serving",
    "whole": "whole",
    "tbsp": "tbsp",
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "tsp": "tsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
}

BASE_UNIT_GRAMS: dict[str, float] = {
    "g": 1,
    "kg": 1000,
    "ml": 1,
    "l": 1000,
    "eggs": 50,
    "whole": 100,
    "piece": 100,
    "slice": 30,
    "cup": 240,
    "bowl": 250,
    "plate": 300,
    "serving": 150,
    "tbsp": 15,
    "tsp": 5,
}

FOOD_SPECIFIC_UNIT_GRAMS: list[tuple[str, str, float]] = [
    ("pizza", "whole", 320),
    ("pizza", "slice", 95),
    ("paneer butter masala", "bowl", 250),
    ("paneer butter masala", "serving", 220),
    ("poha", "bowl", 220),
    ("naan", "piece", 80),
    ("coke", "glass", 250),
]

SYNONYM_CANONICAL = {
    "coca cola": "coke",
    "coca-cola": "coke",
    "cola": "coke",
    "soft drink": "coke",
    "margherita": "margherita pizza",
    "paneer makhani": "paneer butter masala",
}

# Per 100g profiles. Values are approximate and intentionally conservative.
LOCAL_FOOD_DB: list[dict[str, Any]] = [
    {
        "canonical": "boiled egg",
        "aliases": ["egg", "eggs", "boiled egg", "omelette"],
        "kcal_100g": 155,
        "protein_100g": 13,
        "carbs_100g": 1.1,
        "fats_100g": 11,
    },
    {
        "canonical": "poha",
        "aliases": ["poha", "kanda poha", "flattened rice poha"],
        "kcal_100g": 180,
        "protein_100g": 3.5,
        "carbs_100g": 31,
        "fats_100g": 5,
    },
    {
        "canonical": "paneer butter masala",
        "aliases": ["paneer butter masala", "paneer makhani", "butter paneer"],
        "kcal_100g": 220,
        "protein_100g": 9,
        "carbs_100g": 8,
        "fats_100g": 17,
    },
    {
        "canonical": "naan",
        "aliases": ["naan", "butter naan", "plain naan"],
        "kcal_100g": 300,
        "protein_100g": 9,
        "carbs_100g": 50,
        "fats_100g": 7,
    },
    {
        "canonical": "margherita pizza",
        "aliases": ["pizza", "margherita pizza", "cheese pizza", "double cheese margherita pizza"],
        "kcal_100g": 270,
        "protein_100g": 11,
        "carbs_100g": 33,
        "fats_100g": 10,
    },
    {
        "canonical": "coke",
        "aliases": ["coke", "coca cola", "cola", "soft drink"],
        "kcal_100g": 42,
        "protein_100g": 0,
        "carbs_100g": 10.6,
        "fats_100g": 0,
    },
]


@lru_cache(maxsize=512)
def normalize_food_name(name: str) -> str:
    cleaned = re.sub(r"\([^)]*\)", " ", str(name or "").lower())
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", cleaned)
    tokens = [t for t in cleaned.replace("-", " ").split() if t and t not in BRAND_NOISE_TOKENS]
    joined = " ".join(tokens).strip()
    for raw, canon in SYNONYM_CANONICAL.items():
        if raw in joined:
            joined = joined.replace(raw, canon)
    deduped: list[str] = []
    for token in joined.split():
        if not deduped or deduped[-1] != token:
            deduped.append(token)
    joined = " ".join(deduped)
    return re.sub(r"\s+", " ", joined).strip()


def normalize_unit(unit: str | None) -> str:
    if not unit:
        return "serving"
    return UNIT_ALIASES.get(unit.strip().lower(), unit.strip().lower())


def estimate_grams(quantity: float | None, unit: str | None, food_name: str) -> float:
    qty = float(quantity) if quantity not in (None, "") else 1.0
    qty = max(qty, 0.0)
    n_unit = normalize_unit(unit)
    n_name = normalize_food_name(food_name)

    for key, specific_unit, grams in FOOD_SPECIFIC_UNIT_GRAMS:
        if key in n_name and n_unit == specific_unit:
            return qty * grams

    per_unit = BASE_UNIT_GRAMS.get(n_unit, BASE_UNIT_GRAMS["serving"])
    return qty * per_unit


def _energy_from_macros(protein: float, carbs: float, fats: float) -> float:
    return max(0.0, protein * 4 + carbs * 4 + fats * 9)


def _confidence_step_down(level: str) -> str:
    idx = CONFIDENCE_ORDER.index(level) if level in CONFIDENCE_ORDER else 0
    return CONFIDENCE_ORDER[max(0, idx - 1)]


@lru_cache(maxsize=512)
def _best_local_match(normalized_name: str) -> tuple[dict[str, Any] | None, float, str]:
    best_profile: dict[str, Any] | None = None
    best_score = 0.0
    best_type = "none"

    for profile in LOCAL_FOOD_DB:
        aliases = [profile["canonical"], *profile.get("aliases", [])]
        aliases = [normalize_food_name(a) for a in aliases]
        if normalized_name in aliases:
            return profile, 1.0, "exact"
        for alias in aliases:
            if alias and alias in normalized_name:
                return profile, 0.9, "fuzzy"
            score = difflib.SequenceMatcher(None, normalized_name, alias).ratio()
            if score > best_score:
                best_profile = profile
                best_score = score
                best_type = "fuzzy"

    if best_score >= 0.78:
        return best_profile, best_score, best_type
    return None, 0.0, "none"


def _profile_to_item(profile: dict[str, Any], grams: float) -> dict[str, float]:
    factor = grams / 100.0
    protein = float(profile["protein_100g"]) * factor
    carbs = float(profile["carbs_100g"]) * factor
    fats = float(profile["fats_100g"]) * factor
    calories = _energy_from_macros(protein, carbs, fats)
    return {
        "Calories": round(calories),
        "Protein": round(protein, 1),
        "Carbs": round(carbs, 1),
        "Fats": round(fats, 1),
    }


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
    q = re.sub(r"\s+", " ", normalize_food_name(food_name))[:80]
    if len(q) < 2:
        return None
    try:
        response = requests.get(
            OFF_SEARCH,
            params={
                "search_terms": q,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": 6,
            },
            headers={"User-Agent": "NutriVoice/1.0 (Streamlit local app)"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError):
        return None

    products = data.get("products") or []
    best = None
    best_score = -1.0
    for product in products:
        per = _pick_nutriments(product)
        if not per or per["kcal"] <= 0:
            continue
        product_name = normalize_food_name(product.get("product_name") or product.get("product_name_en") or "")
        score = difflib.SequenceMatcher(None, q, product_name).ratio()
        if score > best_score:
            best_score = score
            best = (product, per)

    if not best:
        return None

    product, per = best
    qty = float(quantity) if quantity is not None else 1.0
    grams = estimate_grams(qty, unit, food_name)
    factor = grams / 100.0
    pname = (product.get("product_name") or product.get("product_name_en") or food_name).strip()
    return (
        {
            "Calories": round(per["kcal"] * factor),
            "Protein": round(per["protein"] * factor, 1),
            "Carbs": round(per["carbs"] * factor, 1),
            "Fats": round(per["fats"] * factor, 1),
        },
        pname,
    )


def _sanity_flags(name: str, quantity: float | None, unit: str | None, macros: dict[str, float]) -> list[str]:
    flags: list[str] = []
    n_name = normalize_food_name(name)
    protein = max(0.0, float(macros.get("Protein", 0) or 0))
    carbs = max(0.0, float(macros.get("Carbs", 0) or 0))
    fats = max(0.0, float(macros.get("Fats", 0) or 0))

    if any(k in n_name for k in ("egg", "eggs")) and quantity and normalize_unit(unit) == "eggs":
        per_egg = protein / max(float(quantity), 1)
        if per_egg < 4 or per_egg > 9:
            flags.append("egg_protein_outlier")

    if any(k in n_name for k in ("coke", "cola", "soft drink", "soda")):
        if protein > 1 or fats > 1:
            flags.append("beverage_macro_anomaly")

    if "paneer" in n_name and fats < max(8, protein * 0.7):
        flags.append("paneer_low_fat_anomaly")

    if "pizza" in n_name and fats < 6:
        flags.append("pizza_low_fat_anomaly")

    if protein + carbs + fats <= 0:
        flags.append("missing_macros")

    return flags


def validate_and_correct(item: dict[str, Any]) -> dict[str, Any]:
    protein = max(0.0, float(item.get("Protein", 0) or 0))
    carbs = max(0.0, float(item.get("Carbs", 0) or 0))
    fats = max(0.0, float(item.get("Fats", 0) or 0))
    calories = max(0.0, float(item.get("Calories", 0) or 0))
    confidence = str(item.get("confidence", "medium-low"))
    flags = list(item.get("flags", []))

    calculated = _energy_from_macros(protein, carbs, fats)
    if calories <= 0 and calculated > 0:
        calories = calculated
        flags.append("calories_recomputed_from_macros")

    mismatch = abs(calories - calculated) / max(calories, calculated, 1.0)
    if mismatch > 0.10:
        calories = calculated
        flags.append("corrected_macros")
        flags.append("macro_calorie_mismatch")
        confidence = _confidence_step_down(confidence)

    sanity = _sanity_flags(item.get("Name", ""), item.get("quantity"), item.get("unit"), {
        "Protein": protein,
        "Carbs": carbs,
        "Fats": fats,
    })
    if sanity:
        flags.extend(sanity)
        confidence = _confidence_step_down(confidence)

    source = str(item.get("nutrition_source", "llm_fallback"))
    if source.startswith("llm") and any(f in flags for f in ("corrected_macros", "macro_calorie_mismatch")):
        source = "llm_corrected"

    item.update(
        {
            "food": str(item.get("Name", "")).strip(),
            "Calories": int(round(calories)),
            "Protein": round(protein, 1),
            "Carbs": round(carbs, 1),
            "Fats": round(fats, 1),
            "fat": round(fats, 1),
            "source": str(item.get("source", source)),
            "confidence": confidence,
            "flags": sorted(set(flags)),
            "nutrition_source": source,
            "confidence_status": confidence_to_status(confidence),
        }
    )
    return item


def confidence_to_status(confidence: str) -> str:
    mapping = {
        "high": "matched",
        "medium": "matched",
        "medium-low": "needs_review",
        "low": "fallback",
    }
    return mapping.get(confidence, "needs_review")


def _resolve_entity_from_db(name: str, quantity: float | None, unit: str | None) -> dict[str, Any] | None:
    normalized = normalize_food_name(name)
    grams = estimate_grams(quantity, unit, name)

    profile, score, match_type = _best_local_match(normalized)
    if profile:
        macros = _profile_to_item(profile, grams)
        confidence = "high" if match_type == "exact" else "medium"
        return {
            "Name": profile["canonical"],
            "quantity": quantity,
            "unit": normalize_unit(unit),
            **macros,
            "source": "db_exact" if match_type == "exact" else "db_fuzzy",
            "nutrition_source": "db",
            "confidence": confidence,
            "flags": [] if match_type == "exact" else [f"fuzzy_match_{score:.2f}"],
        }

    off = lookup_open_food_facts(name, quantity, unit)
    if off:
        macros, pname = off
        return {
            "Name": pname or name,
            "quantity": quantity,
            "unit": normalize_unit(unit),
            **macros,
            "source": "db_off",
            "nutrition_source": "db",
            "confidence": "medium",
            "flags": ["open_food_facts_match"],
        }

    return None


def resolve_meal_text(
    text: str, api_key: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Resolve meal text via modular nutrition pipeline."""
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
    fallback_entities: list[dict[str, Any]] = []

    for ent in entities:
        raw_name = str(ent.get("name") or "Unknown")
        qty_raw = ent.get("quantity")
        try:
            quantity = float(qty_raw) if qty_raw is not None else 1.0
        except (TypeError, ValueError):
            quantity = 1.0
        unit = normalize_unit(ent.get("unit"))

        db_item = _resolve_entity_from_db(raw_name, quantity, unit)
        if db_item:
            resolved.append(validate_and_correct(db_item))
            continue

        fallback_entities.append(
            {
                "name": normalize_food_name(raw_name) or raw_name,
                "quantity": quantity,
                "unit": unit,
                "meal_type": ent.get("meal_type"),
            }
        )

    if fallback_entities:
        estimates = utils.estimate_nutrition_batch(fallback_entities, api_key)
        for idx, ent in enumerate(fallback_entities):
            est = estimates[idx] if idx < len(estimates) else {}
            candidate = {
                "Name": str(est.get("Name") or ent["name"]),
                "quantity": float(est.get("quantity") or ent["quantity"] or 1.0),
                "unit": normalize_unit(est.get("unit") or ent["unit"]),
                "Calories": float(est.get("Calories", 0) or 0),
                "Protein": float(est.get("Protein", 0) or 0),
                "Carbs": float(est.get("Carbs", 0) or 0),
                "Fats": float(est.get("Fats", 0) or 0),
                "source": "llm_fallback",
                "nutrition_source": "llm_fallback",
                "confidence": "medium-low",
                "flags": ["llm_fallback"],
            }
            resolved.append(validate_and_correct(candidate))

    return resolved, meal_type


def transcribe_audio_only(audio_bytes: bytes, mime_type: str) -> tuple[str | None, str | None]:
    """Returns (transcript, error_message). No LLM."""
    return utils.transcribe_audio_bytes(audio_bytes, mime_type)
