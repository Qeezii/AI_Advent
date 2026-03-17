"""
Flask-приложение для работы с YandexGPT API
"""

import os
import logging
import json
import time
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация Flask
app = Flask(__name__)

# === Конфигурация YandexGPT ===
FOLDER_ID = os.getenv("FOLDER_ID")
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL_URL")
BASE_URL = "https://ai.api.cloud.yandex.net/v1"

# Валидация конфигурации
if not FOLDER_ID or not API_KEY:
    logger.error("❌ Не настроены FOLDER_ID или API_KEY в .env")
    raise RuntimeError("Проверьте файл .env — необходимы FOLDER_ID и API_KEY")

# Инициализация OpenAI-клиента (Yandex совместим с OpenAI API)
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    project=FOLDER_ID
)

def call_yandexgpt(prompt: str, 
                   system_prompt: str = "Ты инженер по имитационному моделированию.",
                   temperature: float = 0.3,
                   max_tokens: int = 1000,
                   stop_sequences: list[str] | None = None) -> dict:
    """
    Отправляет запрос в YandexGPT и возвращает ответ с метаданными.
    
    :param prompt: Текст запроса от пользователя
    :param system_prompt: Системная инструкция для модели
    :param temperature: Креативность ответа (0.0 - 1.0)
    :param max_tokens: Максимальная длина ответа
    :param stop_sequences: Стоп-последовательности для остановки генерации
    :return: Словарь с ответом и метаданными
    """
    start_time = time.time()

    try:
        # Формирование сообщения в формате OpenAI-compatible
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Параметры запроса
        request_params = {
            "model": f"gpt://{FOLDER_ID}/{MODEL}",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Добавление stop sequences
        if stop_sequences:
            request_params["stop"] = stop_sequences
        
        response = client.chat.completions.create(**request_params)
        
        elapsed_time = time.time() - start_time
        
        return {
            "success": True,
            "text": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "metadata": {
                "max_tokens": max_tokens,
                "stop_sequences": stop_sequences,
                "temperature": temperature,
                "time_seconds": round(elapsed_time, 2)
            }
        }
    except Exception as e:
        logger.error(f"Ошибка при вызове API: {e}")
        return {
            "success": False,
            "text": f"⚠️ Ошибка: {type(e).__name__} - {str(e)}",
            "usage": {},
            "metadata": {}
        }


# Маршруты веб-интерфейса
@app.route('/')
def index():
    """Главная страница с формой чата"""
    return render_template('index.html')

@app.route('/api/compare', methods=['POST'])
def api_compare():
    """
    API-эндпоинт для сравнения ответов с ограничениями и без
    """
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Поле "message" обязательно'}), 400
    
    prompt = data['message'].strip()
    if not prompt:
        return jsonify({'error': 'Сообщение не может быть пустым'}), 400
    
    logger.info(f"📩 Запрос от пользователя: {prompt[:100]}...")
    
    # Запрос без ограничений
    response_no_limits = call_yandexgpt(
        prompt=prompt,
        system_prompt="Ты инженер по имитационному моделированию. Отвечай подробно и развернуто.",
        temperature=0.7,
        max_tokens=1000,  # Большая длина
        stop_sequences=None  # Нет стоп-последовательностей
    )
    
    # Запрос с ограничениями
    # Добавление явного описания формата в системный промпт
    format_instruction = """
    ФОРМАТ ОТВЕТА:
    - Ответ должен быть кратким (не более 3 предложений)
    - Используй структуру: 1) Главный вывод 2) Ключевые факты 3) Рекомендация
    - Заверши ответ словом "END"
    - Не используй маркеры списка
    """
    
    response_with_limits = call_yandexgpt(
        prompt=prompt + "\n\n" + format_instruction,
        system_prompt="Ты инженер по имитационному моделированию. Строго следуй формату ответа.",
        temperature=0.1,  # Меньше креативности
        max_tokens=200,   # Ограничение длины
        stop_sequences=["END", "###", "\n\n\n"]  # Стоп-последовательности
    )
    
    logger.info(f"📤 Ответы отправлены")
    
    return jsonify({
        "prompt": prompt,
        "without_limits": response_no_limits,
        "with_limits": response_with_limits,
        "comparison": {
            "token_difference": (
                response_no_limits["usage"].get("completion_tokens", 0) - 
                response_with_limits["usage"].get("completion_tokens", 0)
            ),
            "length_difference": (
                len(response_no_limits["text"]) - len(response_with_limits["text"])
            )
        }
    })

# Обработчики ошибок
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


# Запуск приложения
if __name__ == '__main__':
    port = int(os.getenv('FLASK_RUN_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"🚀 Запуск сервера на порту {port} (debug={debug})")
    logger.info(f"🔗 Откройте в браузере: http://localhost:{port}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)