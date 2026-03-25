from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import speech_recognition as sr
from groq import Groq
from pydub import AudioSegment

MODEL_NAME = "llama-3.3-70b-versatile"

EXTRACT_SYSTEM_PROMPT = """You extract structured food entities from the user's meal description.
Return ONLY a JSON object with key "items" — an array of objects, each with:
- "name": string, canonical food description (e.g. "boiled eggs", "poha")
- "quantity": number (amount consumed; default 1 if unspecified)
- "unit": string, one of: g, ml, whole, eggs, cup, bowl, tbsp, tsp, slice, piece, serving (use best guess)
- "meal_type": optional string: breakfast, lunch, dinner, snack, or null if unknown

Do not include calories or macros. Be concise.
Example:
{"items":[{"name":"eggs","quantity":2,"unit":"eggs","meal_type":"breakfast"},{"name":"poha","quantity":1,"unit":"bowl","meal_type":"breakfast"}]}
"""

ESTIMATE_BATCH_PROMPT = """You estimate nutrition for each food item. Return ONLY JSON with key "items".
Each item must have: "Name", "quantity", "unit", "Calories" (int), "Protein", "Carbs", "Fats".
Rules:
- Keep values realistic for cooked food portions.
- Ensure calorie consistency: Calories ~= Protein*4 + Carbs*4 + Fats*9.
- Avoid impossible precision: use 1 decimal max for macros.
- Indian rich gravies (e.g., paneer butter masala) are typically fat-heavy.
- Sugary soft drinks (e.g., coke) should be mostly carbs, near-zero protein/fat.
"""


def _groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def _parse_items_array(content: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    items: list[Any] = []
    if isinstance(data, dict):
        if "items" in data:
            items = data["items"]
        elif "foods" in data:
            items = data["foods"]
        else:
            for val in data.values():
                if isinstance(val, list):
                    items = val
                    break
    elif isinstance(data, list):
        items = data
    if not isinstance(items, list):
        return []
    return [i for i in items if isinstance(i, dict)]


def extract_food_entities(text: str, api_key: str) -> list[dict[str, Any]]:
    """LLM: natural language -> list of {name, quantity, unit, meal_type?}."""
    if not api_key or not text.strip():
        return []
    try:
        client = _groq_client(api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            model=MODEL_NAME,
            response_format={"type": "json_object"},
        )
        raw = chat_completion.choices[0].message.content or "{}"
        items = _parse_items_array(raw)
        out: list[dict[str, Any]] = []
        for i in items:
            name = i.get("name") or i.get("Name")
            if not name:
                continue
            out.append(
                {
                    "name": str(name).strip(),
                    "quantity": i.get("quantity", 1),
                    "unit": i.get("unit") or "serving",
                    "meal_type": i.get("meal_type"),
                }
            )
        return out
    except Exception as e:
        print(f"extract_food_entities failed: {e}")
        return []


def estimate_nutrition_batch(
    entities: list[dict[str, Any]], api_key: str
) -> list[dict[str, Any]]:
    """LLM fallback for items that did not match Open Food Facts."""
    if not api_key or not entities:
        return []
    try:
        client = _groq_client(api_key)
        user_payload = json.dumps({"items": entities})
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": ESTIMATE_BATCH_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            model=MODEL_NAME,
            response_format={"type": "json_object"},
        )
        raw = chat_completion.choices[0].message.content or "{}"
        items = _parse_items_array(raw)
        results: list[dict[str, Any]] = []
        for i in items:
            if not isinstance(i, dict):
                continue
            name = i.get("Name") or i.get("name")
            if not name:
                continue
            results.append(
                {
                    "Name": str(name),
                    "quantity": i.get("quantity"),
                    "unit": i.get("unit"),
                    "Calories": int(float(i.get("Calories", 0))),
                    "Protein": float(i.get("Protein", 0)),
                    "Carbs": float(i.get("Carbs", 0)),
                    "Fats": float(i.get("Fats", 0)),
                }
            )
        return results
    except Exception as e:
        print(f"estimate_nutrition_batch failed: {e}")
        return []


def transcribe_audio_bytes(audio_bytes: bytes, mime_type: str) -> tuple[str | None, str | None]:
    """
    Device speech recognition (Google via SpeechRecognition). Returns (text, error).
    """
    ext = ".wav"
    if "webm" in (mime_type or ""):
        ext = ".webm"
    elif "mpeg" in (mime_type or ""):
        ext = ".mp3"

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        if ext != ".wav":
            audio = AudioSegment.from_file(tmp_path)
            wav_path = tmp_path.replace(ext, "_conv.wav")
            audio.export(wav_path, format="wav")
            os.remove(tmp_path)
            tmp_path = wav_path

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
        transcription = recognizer.recognize_google(audio_data)
        return transcription, None
    except sr.UnknownValueError:
        return None, "Could not understand audio. Try again or use text input."
    except sr.RequestError as e:
        return None, f"Speech service error: {e}"
    except Exception as e:
        return None, str(e)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
