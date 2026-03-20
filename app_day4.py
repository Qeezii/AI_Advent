import os
import logging
import time
import re
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Конфигурация YandexGPT
FOLDER_ID = os.getenv("FOLDER_ID")
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL_URL")
BASE_URL = "https://ai.api.cloud.yandex.net/v1"

if not FOLDER_ID or not API_KEY:
    raise RuntimeError("Не заданы FOLDER_ID или API_KEY")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, project=FOLDER_ID)

def call_yandexgpt(prompt, system_prompt="Ты эксперт по металлургии и цифровым двойникам.", temperature=0.7, max_tokens=2000):
    start_time = time.time()
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        params = {
            "model": f"gpt://{FOLDER_ID}/{MODEL}",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = client.chat.completions.create(**params)
        elapsed = time.time() - start_time
        text = response.choices[0].message.content.strip()
        # простые метрики разнообразия
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words) if words else 0
        return {
            "success": True,
            "text": text,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "metadata": {
                "temperature": temperature,
                "time_seconds": round(elapsed, 2),
                "length": len(text),
                "unique_words": len(unique_words),
                "lexical_diversity": round(lexical_diversity, 3)
            }
        }
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return {
            "success": False,
            "text": f"Ошибка: {str(e)}",
            "usage": {},
            "metadata": {"temperature": temperature, "time_seconds": round(time.time() - start_time, 2)}
        }

TASK = "Как в цифровом двойнике (ЦД) спрогнозировать, что сляб будет транспортироваться медленно и его можно выдавать из печей чуть раньше?"

@app.route('/')
def index():
    return render_template('index_day4.html')

@app.route('/api/compare_temperature', methods=['POST'])
def compare_temperature():
    temperatures = [0, 0.5, 1]
    results = {}
    for t in temperatures:
        results[str(t)] = call_yandexgpt(TASK, temperature=t)
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.getenv('FLASK_RUN_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)