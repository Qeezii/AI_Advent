"""
Flask-приложение для работы с YandexGPT API
Режимы:
- unlimited: развёрнутые ответы без ограничений
- restricted: диалоговый сбор данных в JSON с последующей генерацией ТЗ
"""
import os
import logging
import json
import time
from flask import Flask, render_template, request, jsonify, session
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
MODEL = os.getenv("MODEL_URL")  # например "gpt://<folder-id>/yandexgpt/latest"
BASE_URL = "https://ai.api.cloud.yandex.net/v1"

if not FOLDER_ID or not API_KEY:
    logger.error("❌ Не настроены FOLDER_ID или API_KEY в .env")
    raise RuntimeError("Проверьте файл .env")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, project=FOLDER_ID)

# === Системные промпты ===
SYSTEM_PROMPT_UNLIMITED = (
    "Ты инженер по имитационному моделированию и разработке цифровых двойников на производстве. "
    "Отвечай подробно и развёрнуто."
)

SYSTEM_PROMPT_RESTRICTED_DIRECT = (
    "Ты инженер по имитационному моделированию. Отвечай на запрос пользователя кратко, не более 3 предложений. "
    "Используй структуру: 1) Главный вывод 2) Ключевые факты 3) Рекомендация. Заверши ответ словом END. "
    "Не используй маркеры списка."
)

SYSTEM_PROMPT_COLLECTOR = """
Ты инженер по имитационному моделированию и разработке цифровых двойников на производстве.
Твоя задача — собрать все необходимые данные для создания Цифрового Двойника (ЦД) и сгенерировать техническое задание.

ПРАВИЛА РАБОТЫ:
1. Задавай вопросы по одному за раз, чтобы собрать полную информацию.
2. Всегда отвечай ТОЛЬКО в JSON-формате по следующей схеме (без markdown-блоков, только чистый JSON):
{
    "status": "collecting" | "complete",
    "question": "Твой вопрос пользователю" | null,
    "collected_data": {},  // накопленные данные (обновляй при каждом ответе)
    "final_result": {} | null,  // готовое ТЗ, когда status="complete"
    "message": "Краткое сообщение пользователю"  // например, пояснение или итог
}
3. Когда пользователь скажет "хватит", "всё", "готово", "нет больше вопросов" — переводи status в "complete" и заполняй final_result.
4. В конце каждого ответа (кроме финального) обязательно спроси: "Есть ли ещё что-то, что я должна узнать?".
5. Не добавляй никакой текст вне JSON-объекта.
6. Не используй markdown-блоки типа ```json.

Начни с первого вопроса.
"""

# === Конфигурации режимов ===
MODES = {
    "unlimited": {
        "name": "Без ограничений",
        "system_prompt": SYSTEM_PROMPT_UNLIMITED,
        "temperature": 0.7,
        "max_tokens": 2000,
        "stop_sequences": None,
        "description": "Полные развёрнутые ответы"
    },
    "restricted": {
        "name": "С ограничениями (диалог + JSON)",
        "system_prompt": SYSTEM_PROMPT_COLLECTOR,  # используется для диалога
        "temperature": 0.3,
        "max_tokens": 500,
        "stop_sequences": ["END", "###", "\n\n\n"],
        "description": "Диалоговый сбор данных в JSON, итоговое ТЗ"
    }
}

def call_yandexgpt(prompt: str,
                   system_prompt: str | None = None,
                   temperature: float = 0.3,
                   max_tokens: int = 1000,
                   stop_sequences: list[str] | None = None,
                   conversation_history: list[dict] | None = None,
                   mode: str = "unlimited") -> dict:
    """
    Отправляет запрос в YandexGPT и возвращает ответ с метаданными.
    """
    start_time = time.time()
    prompt = (prompt or "").strip()
    if not prompt:
        logger.error("❌ Пустой prompt")
        return {
            "success": False,
            "text": "⚠️ Ошибка: пустой запрос",
            "parsed_json": None,
            "usage": {},
            "metadata": {"mode": mode, "time_seconds": 0}
        }

    # Если системный промпт не передан, берём из режима (но для restricted используем collector)
    if system_prompt is None:
        system_prompt = MODES[mode]["system_prompt"]

    try:
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"📤 Запрос: {len(messages)} сообщений, {len(prompt)} символов")

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

        response_text = response.choices[0].message.content.strip()
        parsed_json = None

        # Очистка от markdown-обёрток
        clean_text = response_text
        if response_text.startswith("```"):
            clean_text = response_text.split("```", 1)[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]
            clean_text = clean_text.rsplit("```", 1)[0].strip()

        try:
            parsed_json = json.loads(clean_text)
        except json.JSONDecodeError:
            pass

        return {
            "success": True,
            "text": response_text,  # всегда возвращаем оригинал
            "parsed_json": parsed_json,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "metadata": {
                "max_tokens": max_tokens,
                "stop_sequences": stop_sequences,
                "temperature": temperature,
                "time_seconds": round(elapsed, 2),
                "mode": mode
            }
        }
    except Exception as e:
        logger.error(f"Ошибка API: {e}")
        return {
            "success": False,
            "text": f"⚠️ Ошибка: {type(e).__name__} - {str(e)}",
            "parsed_json": None,
            "usage": {},
            "metadata": {"mode": mode, "time_seconds": round(time.time() - start_time, 2)}
        }

# === Маршруты ===
@app.route('/')
def index():
    return render_template('index_day2.html', modes=MODES)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Обычный чат с учётом режима. В restricted режиме работает диалоговый сбор."""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Поле "message" обязательно'}), 400

    prompt = data['message'].strip()
    if not prompt:
        return jsonify({'error': 'Сообщение не может быть пустым'}), 400

    mode = data.get('mode', 'unlimited')
    if mode not in MODES:
        mode = 'unlimited'

    logger.info(f"📩 Запрос (режим: {mode}): {prompt[:100]}...")

    # Получаем историю из сессии
    conversation_history = session.get('conversation_history', [])

    mode_config = MODES[mode]
    # Для restricted используем системный промпт-коллектор (уже в MODES)
    system_prompt = mode_config["system_prompt"]

    response = call_yandexgpt(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=mode_config["temperature"],
        max_tokens=mode_config["max_tokens"],
        stop_sequences=mode_config["stop_sequences"],
        conversation_history=conversation_history[-20:],  # ограничим историю
        mode=mode
    )

    # Обновляем историю
    conversation_history.append({"role": "user", "content": prompt})
    if response['success']:
        conversation_history.append({"role": "assistant", "content": response['text']})
    else:
        conversation_history.append({"role": "assistant", "content": response['text']})  # показываем ошибку

    session['conversation_history'] = conversation_history

    return jsonify({
        "prompt": prompt,
        "response": response['text'],
        "parsed_json": response['parsed_json'],
        "usage": response['usage'],
        "metadata": response['metadata'],
        "mode": mode,
        "mode_name": MODES[mode]["name"]
    })

@app.route('/api/compare', methods=['POST'])
def api_compare():
    """
    Сравнение ответов на один запрос в двух режимах:
    - без ограничений (unlimited, прямой ответ)
    - с ограничениями (restricted, прямой ответ без диалога, с JSON)
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Поле "message" обязательно'}), 400

    prompt = data['message'].strip()
    if not prompt:
        return jsonify({'error': 'Сообщение не может быть пустым'}), 400

    logger.info(f"📩 Запрос на сравнение: {prompt[:100]}...")

    # Режим без ограничений
    unlimited_resp = call_yandexgpt(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT_UNLIMITED,
        temperature=0.7,
        max_tokens=2000,
        stop_sequences=None,
        mode="unlimited"
    )

    # Режим с ограничениями (прямой ответ, без диалога)
    restricted_resp = call_yandexgpt(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT_RESTRICTED_DIRECT,
        temperature=0.1,
        max_tokens=500,
        stop_sequences=["END", "###", "\n\n\n"],
        mode="restricted"
    )

    return jsonify({
        "prompt": prompt,
        "without_limits": unlimited_resp,
        "with_limits": restricted_resp,
        "comparison": {
            "token_difference": (
                unlimited_resp["usage"].get("completion_tokens", 0) -
                restricted_resp["usage"].get("completion_tokens", 0)
            ),
            "length_difference": (
                len(unlimited_resp["text"]) - len(restricted_resp["text"])
            )
        }
    })

@app.route('/api/reset', methods=['POST'])
def api_reset():
    session.pop('conversation_history', None)
    return jsonify({"success": True, "message": "Диалог сброшен"})

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('FLASK_RUN_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    logger.info(f"🚀 Запуск на порту {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)