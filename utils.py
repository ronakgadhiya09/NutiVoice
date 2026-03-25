from groq import Groq
import json
import os
import speech_recognition as sr
import tempfile
from pydub import AudioSegment

MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert nutritionist and calorie estimator.
Your task is to analyze the user's input and return a JSON object containing a list called "items".
Each object in the "items" list MUST represent a food item extracted from the input and include:
- "Name": (string) Name of the food.
- "Calories": (integer) Estimated calories.
- "Protein": (integer) Estimated protein in grams.
- "Carbs": (integer) Estimated carbohydrates in grams.
- "Fats": (integer) Estimated fats in grams.

Respond ONLY with the JSON object.
Example response:
{
  "items": [
    {"Name": "2 Eggs", "Calories": 140, "Protein": 12, "Carbs": 1, "Fats": 10},
    {"Name": "Bowl of Poha", "Calories": 250, "Protein": 5, "Carbs": 45, "Fats": 6}
  ]
}
"""

def parse_food_text(text: str, api_key: str) -> list:
    """Uses Groq Llama-3.3 to parse food items from standard text."""
    try:
        if not api_key:
            return []
        print(f"DEBUG: Using model llama-3.3-70b-versatile for text: {text[:50]}...")
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"}
        )
        
        data = json.loads(chat_completion.choices[0].message.content)
        items = []
        if isinstance(data, dict):
            if "items" in data: items = data["items"]
            elif "foods" in data: items = data["foods"]
            elif "Name" in data: items = [data]
            else:
                for val in data.values():
                    if isinstance(val, list):
                        items = val
                        break
        elif isinstance(data, list):
            items = data
            
        # EXTRA ROBUSTNESS: Ensure every item is actually a dictionary
        if isinstance(items, list):
            return [i for i in items if isinstance(i, dict)]
        return []
    except Exception as e:
        print(f"Failed to parse text via Groq: {e}")
        return []

def parse_food_audio(audio_bytes: bytes, mime_type: str, api_key: str) -> list:
    """Uses SpeechRecognition library for transcription, then parses via Groq."""
    try:
        # Save to temp file
        ext = ".wav"
        if "webm" in mime_type: ext = ".webm"
        elif "mpeg" in mime_type: ext = ".mp3"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            # If not a wav, convert to wav using pydub for SpeechRecognition
            if ext != ".wav":
                audio = AudioSegment.from_file(tmp_path)
                wav_path = tmp_path.replace(ext, "_conv.wav")
                audio.export(wav_path, format="wav")
                os.remove(tmp_path)
                tmp_path = wav_path

            recognizer = sr.Recognizer()
            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)
                # Using Google's free recognizer via the SpeechRecognition library
                transcription = recognizer.recognize_google(audio_data)
                
            print(f"Transcript: {transcription}")
            return parse_food_text(transcription, api_key)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        print(f"Failed to parse audio via SpeechRecognition: {e}")
        return []
