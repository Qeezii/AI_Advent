"""
Flask-приложение для работы с YandexGPT API
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, stream_with_context
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
                   max_tokens: int = 1000) -> str:
    """
    Отправляет запрос в YandexGPT и возвращает текстовый ответ.
    
    :param prompt: Текст запроса от пользователя
    :param system_prompt: Системная инструкция для модели
    :param temperature: Креативность ответа (0.0 - 1.0)
    :param max_tokens: Максимальная длина ответа
    :return: Ответ модели или сообщение об ошибке
    """
    try:
        response = client.responses.create(
            model=f"gpt://{FOLDER_ID}/{MODEL}",
            temperature=temperature,
            instructions=system_prompt,
            input=prompt,
            max_output_tokens=max_tokens
        )
        return response.output_text
    except Exception as e:
        logger.error(f"Ошибка при вызове API: {e}")
        return f"⚠️ Ошибка соединения с YandexGPT: {type(e).__name__}"


# Маршруты веб-интерфейса
@app.route('/')
def index():
    """Главная страница с формой чата"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API-эндпоинт для обработки запросов от фронтенда"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Поле "message" обязательно'}), 400
    
    prompt = data['message'].strip()
    if not prompt:
        return jsonify({'error': 'Сообщение не может быть пустым'}), 400
    
    # Параметры из запроса или дефолтные
    temperature = float(data.get('temperature', 0.3))
    max_tokens = int(data.get('max_tokens', 1000))
    system_prompt = data.get('system_prompt', 'Ты полезный ассистент.')
    
    logger.info(f"📩 Запрос от пользователя: {prompt[:100]}...")
    
    # Вызов модели
    response_text = call_yandexgpt(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    logger.info(f"📤 Ответ отправлен: {len(response_text)} символов")
    
    return jsonify({
        'response': response_text,
        'model': MODEL,
        'usage': {
            'prompt_tokens': len(prompt) // 4,  # примерная оценка
            'completion_tokens': len(response_text) // 4
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