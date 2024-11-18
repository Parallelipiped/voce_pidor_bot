# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import warnings
import numpy as np
import torch
import whisper
import soundfile as sf
import codecs
import wave
import shutil
from typing import Optional
from contextlib import contextmanager

# Настраиваем кодировку для stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Игнорируем предупреждения
warnings.filterwarnings("ignore")

@contextmanager
def open_wav_file(path: str, mode: str = 'rb'):
    """Безопасно открывает WAV файл с помощью контекстного менеджера"""
    file = None
    try:
        file = wave.open(path, mode)
        yield file
    finally:
        if file:
            file.close()

def create_temp_copy(path: str) -> Optional[str]:
    """Создает временную копию файла для безопасной обработки"""
    try:
        temp_path = path + '.processing'
        shutil.copy2(path, temp_path)
        return temp_path
    except Exception as e:
        print(f"Не удалось создать временную копию файла: {str(e)}", file=sys.stderr)
        return None

def check_file(path: str) -> bool:
    """Проверяет существование и доступность файла"""
    if not os.path.exists(path):
        print(f"Файл не найден: {path}", file=sys.stderr)
        return False
    
    if not os.path.isfile(path):
        print(f"Путь не является файлом: {path}", file=sys.stderr)
        return False
    
    try:
        with open(path, 'rb') as f:
            # Пробуем прочитать начало файла
            f.read(1024)
        return True
    except Exception as e:
        print(f"Ошибка доступа к файлу {path}: {str(e)}", file=sys.stderr)
        return False

def check_wav_file(path: str) -> bool:
    """Проверяет WAV файл на корректность"""
    try:
        path = os.path.abspath(path)
        print(f"Проверяю файл: {path}", file=sys.stderr)
        
        if not os.path.exists(path):
            print(f"Файл не найден: {path}", file=sys.stderr)
            return False
        
        if not os.path.isfile(path):
            print(f"Путь не является файлом: {path}", file=sys.stderr)
            return False
        
        # Проверяем размер файла
        size = os.path.getsize(path)
        if size == 0:
            print(f"Файл пустой: {path}", file=sys.stderr)
            return False
        print(f"Размер файла: {size} байт", file=sys.stderr)
        
        try:
            with open_wav_file(path) as wav:
                # Проверяем параметры WAV файла
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                rate = wav.getframerate()
                frames = wav.getnframes()
                
                print(f"WAV параметры:", file=sys.stderr)
                print(f"- Каналы: {channels}", file=sys.stderr)
                print(f"- Битность: {width * 8} бит", file=sys.stderr)
                print(f"- Частота: {rate} Гц", file=sys.stderr)
                print(f"- Фреймов: {frames}", file=sys.stderr)
                print(f"- Длительность: {frames / rate:.2f} сек", file=sys.stderr)
                
                if channels != 1:
                    print(f"Неверное количество каналов: {channels}", file=sys.stderr)
                    return False
                if width != 2:  # 16-bit
                    print(f"Неверная битность: {width * 8}", file=sys.stderr)
                    return False
                if rate != 16000:
                    print(f"Неверная частота дискретизации: {rate}", file=sys.stderr)
                    return False
            return True
        except Exception as e:
            print(f"Ошибка при проверке WAV файла {path}: {str(e)}", file=sys.stderr)
            return False
            
    except Exception as e:
        print(f"Ошибка при проверке файла {path}: {str(e)}", file=sys.stderr)
        return False

def check_audio_file(audio_path: str) -> None:
    """Проверяет WAV файл и его параметры."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Файл не найден: {audio_path}")
    
    print(f"Проверяю файл: {audio_path}", file=sys.stderr)
    
    # Получаем информацию о файле
    file_size = os.path.getsize(audio_path)
    print(f"Размер файла: {file_size} байт", file=sys.stderr)
    
    # Читаем WAV параметры
    with sf.SoundFile(audio_path) as f:
        print("WAV параметры:", file=sys.stderr)
        print(f"- Каналы: {f.channels}", file=sys.stderr)
        print(f"- Битность: {f.subtype}", file=sys.stderr)
        print(f"- Частота: {f.samplerate} Гц", file=sys.stderr)
        print(f"- Фреймов: {f.frames}", file=sys.stderr)
        print(f"- Длительность: {f.frames / f.samplerate:.2f} сек", file=sys.stderr)

def load_audio(audio_path: str) -> np.ndarray:
    """Загружает аудио файл в память."""
    audio, _ = sf.read(audio_path, dtype='float32')
    return audio

def format_sentence(text: str) -> str:
    """Форматирует одно предложение"""
    if not text:
        return ""
    # Убираем пробелы в начале и конце
    text = text.strip()
    # Делаем первую букву заглавной
    text = text[0].upper() + text[1:] if text else ""
    # Добавляем точку в конце если нет знаков препинания
    if not text[-1] in '.!?':
        text += '.'
    return text

def format_text(text: str) -> str:
    """Форматирует текст для лучшей читаемости"""
    # Заменяем частые слова-паразиты
    filler_words = {
        'короче': 'Короче говоря,',
        'вауля': 'вуаля',
        'вот': '',
    }
    
    # Слова, требующие запятой перед ними
    comma_words = [
        'который', 'которая', 'которое', 'которые',
        'где', 'куда', 'откуда', 'когда', 'пока',
        'если', 'чтобы', 'потому', 'поэтому',
        'как', 'будто', 'словно', 'точно',
        'что', 'чем', 'хотя', 'пусть'
    ]
    
    # Разбиваем текст на предложения
    sentences = []
    current = []
    
    # Разбиваем на слова и обрабатываем каждое
    words = text.split()
    for i, word in enumerate(words):
        # Пропускаем пустые слова
        if not word:
            continue
            
        # Добавляем запятую перед определенными словами
        if word.lower() in comma_words and current:
            if not current[-1].endswith(('.', ',', '!', '?', ':', ';')):
                current[-1] = current[-1] + ','
                
        # Заменяем слова-паразиты
        word_lower = word.lower()
        if word_lower in filler_words:
            if filler_words[word_lower]:
                current.append(filler_words[word_lower])
            continue
            
        # Добавляем слово
        current.append(word)
        
        # Если слово заканчивается на знак препинания, начинаем новое предложение
        if word.endswith(('.', '!', '?')):
            if current:
                sentences.append(format_sentence(' '.join(current)))
                current = []
                
        # Если следующее слово начинает новое предложение
        elif i < len(words) - 1:
            next_word = words[i + 1].lower()
            if next_word in ['а', 'но', 'и', 'или']:
                if current:
                    sentences.append(format_sentence(' '.join(current) + '.'))
                    current = []
    
    # Добавляем последнее предложение
    if current:
        sentences.append(format_sentence(' '.join(current)))
    
    # Собираем предложения с двумя пробелами между ними
    return '  '.join(sentences)

def clean_text(text: str) -> str:
    """Очищает текст от HTML тегов и специальных символов"""
    # Заменяем <br> и </br> на перенос строки
    text = text.replace('<br>', '\n').replace('</br>', '\n')
    # Заменяем множественные переносы строк на один
    text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    # Форматируем текст
    text = format_text(text)
    return text

def load_config() -> dict:
    """Загружает конфигурацию из JSON файла."""
    config_path = os.path.join(os.path.dirname(sys.argv[1]), "config.json")
    print(f"Looking for config at: {config_path}", file=sys.stderr)
    
    default_config = {
        "speech": {
            "model": "medium",
            "default_language": "ru",
            "languages": {
                "ru": "Russian",
                "en": "English",
                "uk": "Ukrainian",
                "be": "Belarusian"
            },
            "use_gpu": True,
            "auto_detect_language": False
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "bit_depth": 16
        }
    }
    
    if not os.path.exists(config_path):
        print(f"Config file not found at: {config_path}", file=sys.stderr)
        print("Using default configuration", file=sys.stderr)
        return default_config
    
    try:
        print("Loading configuration from file", file=sys.stderr)
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Объединяем с дефолтными значениями
        if "speech" not in config:
            config["speech"] = default_config["speech"]
        else:
            for key, value in default_config["speech"].items():
                if key not in config["speech"] or config["speech"][key] is None:
                    config["speech"][key] = value
                    
        if "audio" not in config:
            config["audio"] = default_config["audio"]
        else:
            for key, value in default_config["audio"].items():
                if key not in config["audio"] or config["audio"][key] is None:
                    config["audio"][key] = value
                    
        print(f"Loaded config: {config}", file=sys.stderr)
        return config
        
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        print("Using default configuration", file=sys.stderr)
        return default_config

def detect_language(audio: np.ndarray, model) -> str:
    """Определяет язык аудио с помощью Whisper."""
    # Используем первые 30 секунд для определения языка
    audio_segment = audio[:int(16000 * 30)] if len(audio) > 16000 * 30 else audio
    
    print("Detecting language...", file=sys.stderr)
    result = model.transcribe(audio_segment, language=None)
    detected_lang = result.get("language", "ru")
    print(f"Detected language: {detected_lang}", file=sys.stderr)
    
    return detected_lang

def transcribe_audio(audio_path: str) -> str:
    try:
        # Начинаем отсчет времени
        start_time = time.time()

        # Проверяем входной файл
        print(f"Начинаю распознавание файла: {audio_path}", file=sys.stderr)
        check_audio_file(audio_path)

        # Загружаем аудио в память
        print("Loading audio file into memory...", file=sys.stderr)
        audio = load_audio(audio_path)
        print(f"Audio loaded: shape={audio.shape}, dtype={audio.dtype}", file=sys.stderr)

        # Определяем устройство
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Используется устройство: {device}", file=sys.stderr)
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
        
        # Загружаем конфигурацию
        config = load_config()
        speech_config = config.get("speech", {})
        
        # Загружаем модель
        print("Loading model...", file=sys.stderr)
        print("Это может занять несколько минут при первом запуске", file=sys.stderr)
        model = whisper.load_model(speech_config.get("model", "medium")).to(device)
        model_load_time = time.time() - start_time
        
        # Определяем язык если включено автоопределение
        language = speech_config.get("default_language", "ru")
        if speech_config.get("auto_detect_language", False):
            language = detect_language(audio, model)
            print(f"Using detected language: {language}", file=sys.stderr)
        else:
            print(f"Using configured language: {language}", file=sys.stderr)
        
        # Распознаем речь
        print(f"Transcribing audio data...", file=sys.stderr)
        transcribe_start = time.time()
        result = model.transcribe(audio, language=language)
        transcribe_time = time.time() - transcribe_start

        # Очищаем и форматируем результат
        text = clean_text(result["text"].strip())
        
        # Освобождаем память GPU
        if device == "cuda":
            torch.cuda.empty_cache()

        # Выводим статистику в stderr
        total_time = time.time() - start_time
        stats = f"\n\n📊 Статистика:\n⏱ Загрузка модели: {model_load_time:.1f}с\n⌛️ Распознавание: {transcribe_time:.1f}с\n🕐 Общее время: {total_time:.1f}с"
        print(f"Распознанный текст:\n{text}", file=sys.stderr)
        print(f"Статистика:{stats}", file=sys.stderr)
        
        # Выводим текст в stdout для Go
        print(text)
        sys.stdout.flush()
        
        return text

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        raise RuntimeError(f"Ошибка при распознавании речи: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python speech_recognition.py <audio_file>", file=sys.stderr)
        sys.exit(1)
    
    try:
        text = transcribe_audio(sys.argv[1])
        sys.exit(0)
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
