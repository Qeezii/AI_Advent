"""
Flask-приложение для сравнения методов решения задачи через YandexGPT.
"""
import os
import logging
import json
import time
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

# === Конфигурация YandexGPT ===
FOLDER_ID = os.getenv("FOLDER_ID")
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL_URL")  # например "yandexgpt/latest"
BASE_URL = "https://ai.api.cloud.yandex.net/v1"

if not FOLDER_ID or not API_KEY:
    logger.error("❌ Не настроены FOLDER_ID или API_KEY в .env")
    raise RuntimeError("Проверьте файл .env")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, project=FOLDER_ID)

# === Базовая функция вызова YandexGPT ===
def call_yandexgpt(prompt: str,
                   system_prompt: str = "Ты эксперт по цифровым двойникам и металлургии.",
                   temperature: float = 0.3,
                   max_tokens: int = 2000,
                   stop_sequences: list[str] | None = None) -> dict:
    """
    Отправляет запрос и возвращает ответ с метаданными.
    """
    start_time = time.time()
    prompt = (prompt or "").strip()
    if not prompt:
        return {"success": False, "text": "⚠️ Пустой запрос", "usage": {}, "metadata": {}}

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
        if stop_sequences:
            params["stop"] = stop_sequences

        response = client.chat.completions.create(**params)
        elapsed = time.time() - start_time

        return {
            "success": True,
            "text": response.choices[0].message.content.strip(),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "metadata": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "time_seconds": round(elapsed, 2)
            }
        }
    except Exception as e:
        logger.error(f"Ошибка API: {e}")
        return {
            "success": False,
            "text": f"⚠️ Ошибка: {type(e).__name__} - {str(e)}",
            "usage": {},
            "metadata": {"time_seconds": round(time.time() - start_time, 2)}
        }

# === Задача для сравнения ===
TASK = "Как в цифровом двойнике (ЦД) спрогнозировать, что сляб будет транспортироваться медленно и его можно выдавать из печей чуть раньше?"

# === Четыре метода ===
def method_direct():
    """Прямой ответ без дополнительных инструкций."""
    return call_yandexgpt(TASK, system_prompt="Ты эксперт по металлургии и цифровым двойникам.")

def method_step_by_step():
    """С инструкцией «решай пошагово»."""
    prompt = f"{TASK}\n\nРешай пошагово, подробно объясняя каждый шаг."
    return call_yandexgpt(prompt, system_prompt="Ты эксперт по металлургии и цифровым двойникам.")

def method_generate_prompt():
    """
    Сначала просим модель составить промпт для решения задачи,
    затем используем этот промпт для получения ответа.
    """
    # Шаг 1: генерация промпта
    gen_prompt = (
        f"Составь подробный промпт (инструкцию) для решения следующей задачи. "
        f"Промпт должен содержать все необходимые шаги и контекст, чтобы другой эксперт мог дать качественный ответ.\n\n"
        f"Задача: {TASK}\n\n"
        f"Верни только текст промпта, без лишних пояснений."
    )
    prompt_response = call_yandexgpt(gen_prompt, system_prompt="Ты методист, составляешь инструкции.")
    if not prompt_response["success"]:
        return prompt_response

    generated_prompt = prompt_response["text"]
    logger.info(f"Сгенерированный промпт: {generated_prompt[:200]}...")

    # Шаг 2: использование сгенерированного промпта для решения задачи
    solution = call_yandexgpt(generated_prompt, system_prompt="Ты эксперт по металлургии и цифровым двойникам.")
    # Добавляем в метаданные сгенерированный промпт для информации
    solution["metadata"]["generated_prompt"] = generated_prompt
    return solution

def method_expert_panel():
    """
    Группа экспертов: аналитик, инженер, критик.
    В одном запросе просим выдать мнения каждого.
    """
    prompt = (
        f"Задача: {TASK}\n\n"
        f"Представь, что ты группа экспертов: Аналитик данных, Инженер-технолог и Критик (скептик). "
        f"Каждый эксперт даёт своё решение задачи. "
        f"Оформи ответ в виде:\n"
        f"**Аналитик:** ...\n"
        f"**Инженер:** ...\n"
        f"**Критик:** ...\n"
        f"Затем, если нужно, можно добавить общий вывод."
    )
    return call_yandexgpt(prompt, system_prompt="Ты координатор группы экспертов.", temperature=0.5)

# === Маршруты ===
@app.route('/')
def index():
    return render_template('index_day3.html')

@app.route('/api/compare_methods', methods=['POST'])
def compare_methods():
    """Запускает все четыре метода и возвращает результаты."""
    logger.info("🔍 Запуск сравнения методов...")
    results = {
        "task": TASK,
        "direct": method_direct(),
        "step_by_step": method_step_by_step(),
        "generated_prompt": method_generate_prompt(),
        "expert_panel": method_expert_panel()
    }
    # Можно добавить простую оценку (например, длину ответа) для сравнения
    for name, res in results.items():
        if name != "task" and res["success"]:
            res["analysis"] = {
                "length": len(res["text"]),
                "tokens": res["usage"].get("completion_tokens", 0)
            }
    return jsonify(results)

@app.route('/api/reset', methods=['POST'])
def api_reset():
    # Заглушка для совместимости, если нужно
    return jsonify({"success": True})

if __name__ == '__main__':
    port = int(os.getenv('FLASK_RUN_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)