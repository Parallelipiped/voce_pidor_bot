# 🎤 VoicePidorBot (Войсоблять)

Telegram бот для распознавания голосовых сообщений с использованием OpenAI Whisper. Бот автоматически конвертирует голосовые сообщения в текст с высокой точностью благодаря использованию модели Whisper и GPU-ускорения.

## ✨ Особенности
- 🎯 Высокая точность распознавания речи
- 🚀 GPU-ускорение (NVIDIA CUDA)
- ⚡ Быстрая обработка (около 20 секунд на сообщение)
- 🔄 Поддержка длинных голосовых сообщений
- 🇷🇺 Оптимизация для русского языка
- 📊 Подробная статистика обработки

## 🛠 Технические требования
- Python 3.10+
- Go 1.21+
- NVIDIA GPU с поддержкой CUDA (опционально)
- FFmpeg
- 4GB+ RAM
- ~2GB свободного места (для моделей)

## 📦 Установка

### 1. Клонирование репозитория
```bash
git clone https://github.com/Parallelipiped/voce_pidor_bot.git
cd voce_pidor_bot
```

### 2. Установка Python зависимостей
```bash
pip install -r requirements.txt
```

### 3. Установка Go зависимостей
```bash
go mod download
```

### 4. Настройка конфигурации
1. Скопируйте `config.example.yaml` в `config.yaml`
2. Отредактируйте `config.yaml`:
   - Добавьте токен вашего Telegram бота (получите у [@BotFather](https://t.me/BotFather))
   - Настройте пути к FFmpeg и временной директории
   - Выберите модель Whisper и параметры распознавания

## 🚀 Запуск
```bash
go run src/go/main.go
```

## ⚙️ Конфигурация
Все настройки хранятся в `config.yaml`:

```yaml
# Telegram Bot Configuration
telegram:
  token: "YOUR_BOT_TOKEN"  # Токен от @BotFather
  max_voice_duration: 1200  # Максимальная длительность сообщения (в секундах)

# Paths Configuration
paths:
  temp_dir: "temp"  # Директория для временных файлов
  python_script: "src/python/speech_recognition.py"
  ffmpeg: "bin/ffmpeg/bin/ffmpeg.exe"

# Speech Recognition Configuration
speech:
  model: "medium"  # Размер модели (tiny, base, small, medium, large)
  language: "ru"   # Язык распознавания
  use_gpu: true    # Использование GPU

# Audio Configuration
audio:
  sample_rate: 16000  # Частота дискретизации (Hz)
  channels: 1         # Количество каналов
  bit_depth: 16       # Битность
```

## 📊 Производительность
Тесты на NVIDIA RTX 3090:
- Загрузка модели: ~9 секунд
- Распознавание: ~11 секунд
- Общее время: ~20 секунд

## 🗂 Структура проекта
```
voce_pidor_bot/
├── bin/
│   └── ffmpeg/          # FFmpeg бинарные файлы
├── src/
│   ├── go/
│   │   └── main.go      # Основной код бота
│   └── python/
│       └── speech_recognition.py  # Код распознавания речи
├── temp/                # Временные файлы
├── config.yaml         # Конфигурация (не включена в репозиторий)
├── config.example.yaml # Пример конфигурации
├── go.mod             # Go зависимости
├── go.sum             # Go зависимости (lock)
└── requirements.txt   # Python зависимости
```

## 📦 Репозитории
- GitHub: [github.com/Parallelipiped/voce_pidor_bot](https://github.com/Parallelipiped/voce_pidor_bot)
- GitVerse: [gitverse.ru/Parallelipiped/voce_pidor_bot](https://gitverse.ru/Parallelipiped/voce_pidor_bot)

## 🤝 Вклад в проект
1. Форкните репозиторий
2. Создайте ветку для ваших изменений
3. Внесите изменения и создайте коммиты
4. Отправьте пулл-реквест

## 📝 Лицензия
MIT License. Подробности в файле [LICENSE](LICENSE).

## ✍️ Автор
- GitHub: [@Parallelipiped](https://github.com/Parallelipiped)
- Telegram: [@Parallelipiped](https://t.me/Parallelipiped)

## 🙏 Благодарности
- [OpenAI Whisper](https://github.com/openai/whisper) за модель распознавания речи
- [FFmpeg](https://ffmpeg.org/) за обработку аудио
- [Go Telegram Bot API](https://github.com/go-telegram-bot-api/telegram-bot-api) за Telegram интеграцию
