package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"
	"gopkg.in/yaml.v3"
)

// Config структура для конфигурации
type Config struct {
	Telegram struct {
		Token            string `yaml:"token"`
		MaxVoiceDuration int    `yaml:"max_voice_duration"`
	} `yaml:"telegram"`
	Paths struct {
		TempDir      string `yaml:"temp_dir"`
		PythonScript string `yaml:"python_script"`
		FFmpeg       string `yaml:"ffmpeg"`
	} `yaml:"paths"`
	Speech struct {
		Model             string            `yaml:"model"`
		DefaultLanguage   string            `yaml:"default_language"`
		Languages         map[string]string `yaml:"languages"`
		UseGPU           bool              `yaml:"use_gpu"`
		AutoDetectLanguage bool             `yaml:"auto_detect_language"`
	} `yaml:"speech"`
	Audio struct {
		SampleRate int `yaml:"sample_rate"`
		Channels   int `yaml:"channels"`
		BitDepth   int `yaml:"bit_depth"`
	} `yaml:"audio"`
}

var config Config

func loadConfig() error {
	// Читаем файл конфигурации
	data, err := os.ReadFile("config.yaml")
	if err != nil {
		return fmt.Errorf("ошибка чтения конфига: %v", err)
	}

	// Парсим YAML
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return fmt.Errorf("ошибка парсинга конфига: %v", err)
	}

	// Получаем абсолютные пути
	projectRoot, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("ошибка получения рабочей директории: %v", err)
	}

	// Преобразуем относительные пути в абсолютные
	if !filepath.IsAbs(config.Paths.TempDir) {
		config.Paths.TempDir = filepath.Join(projectRoot, config.Paths.TempDir)
	}
	if !filepath.IsAbs(config.Paths.PythonScript) {
		config.Paths.PythonScript = filepath.Join(projectRoot, config.Paths.PythonScript)
	}
	if !filepath.IsAbs(config.Paths.FFmpeg) {
		config.Paths.FFmpeg = filepath.Join(projectRoot, config.Paths.FFmpeg)
	}

	// Логируем пути для отладки
	log.Printf("Loaded paths:")
	log.Printf("- TempDir: %s", config.Paths.TempDir)
	log.Printf("- PythonScript: %s", config.Paths.PythonScript)
	log.Printf("- FFmpeg: %s", config.Paths.FFmpeg)

	return nil
}

func createPythonConfig() error {
	// Создаем структуру для Python конфига
	pythonConfig := map[string]interface{}{
		"speech": map[string]interface{}{
			"model":              config.Speech.Model,
			"default_language":   config.Speech.DefaultLanguage,
			"languages":          config.Speech.Languages,
			"use_gpu":           config.Speech.UseGPU,
			"auto_detect_language": config.Speech.AutoDetectLanguage,
		},
		"audio": map[string]interface{}{
			"sample_rate": config.Audio.SampleRate,
			"channels":    config.Audio.Channels,
			"bit_depth":   config.Audio.BitDepth,
		},
	}

	// Преобразуем в JSON
	jsonData, err := json.MarshalIndent(pythonConfig, "", "  ")
	if err != nil {
		return fmt.Errorf("ошибка создания JSON конфига: %v", err)
	}

	// Записываем во временный файл
	configPath := filepath.Join(config.Paths.TempDir, "config.json")
	log.Printf("Creating Python config at: %s", configPath)
	
	err = os.WriteFile(configPath, jsonData, 0644)
	if err != nil {
		return fmt.Errorf("ошибка записи JSON конфига: %v", err)
	}

	// Проверяем, что файл создался
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return fmt.Errorf("конфиг не был создан: %v", err)
	}

	log.Printf("Python config created successfully")
	return nil
}

func init() {
	// Загружаем конфигурацию
	if err := loadConfig(); err != nil {
		log.Fatal("Ошибка загрузки конфигурации:", err)
	}

	// Создаем временную директорию
	if err := os.MkdirAll(config.Paths.TempDir, os.ModePerm); err != nil {
		log.Fatal("Не удалось создать временную директорию:", err)
	}

	// Создаем конфиг для Python
	if err := createPythonConfig(); err != nil {
		log.Fatal("Ошибка создания конфига для Python:", err)
	}

	// Проверяем доступ к временной директории
	testFile := filepath.Join(config.Paths.TempDir, "test.txt")
	if err := os.WriteFile(testFile, []byte("test"), 0666); err != nil {
		log.Fatal("Не удалось записать в временную директорию:", err)
	}
	os.Remove(testFile)
}

func downloadFile(filepath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func convertOggToWav(oggPath string, wavPath string) error {
	// Проверяем входной файл
	if _, err := os.Stat(oggPath); os.IsNotExist(err) {
		return fmt.Errorf("OGG файл не существует: %s", oggPath)
	}

	// Конвертируем OGG в WAV с нужными параметрами для Whisper
	cmd := exec.Command(config.Paths.FFmpeg,
		"-i", oggPath,           // входной файл
		"-ar", "16000",         // частота дискретизации 16kHz
		"-ac", "1",             // моно
		"-c:a", "pcm_s16le",    // 16-bit PCM
		"-y",                    // перезаписать файл если существует
		wavPath,                 // выходной файл
	)

	// Запускаем команду
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("ошибка FFmpeg: %v\nOutput: %s", err, string(output))
	}

	// Проверяем, что файл создан и имеет размер больше 0
	if info, err := os.Stat(wavPath); os.IsNotExist(err) {
		return fmt.Errorf("FFmpeg не создал выходной файл")
	} else if info.Size() == 0 {
		return fmt.Errorf("FFmpeg создал пустой файл")
	}

	return nil
}

func recognizeSpeech(wavPath string) (string, error) {
	// Проверяем существование файла перед запуском Python
	if _, err := os.Stat(wavPath); os.IsNotExist(err) {
		return "", fmt.Errorf("WAV файл не существует перед запуском Python: %s", wavPath)
	}

	// Обновляем конфиг перед каждым распознаванием
	if err := createPythonConfig(); err != nil {
		return "", fmt.Errorf("ошибка создания конфига для Python: %v", err)
	}

	// Запускаем Python скрипт с правильной кодировкой
	cmd := exec.Command("C:\\Users\\serve\\AppData\\Local\\Programs\\Python\\Python310\\python.exe", "-X", "utf8", config.Paths.PythonScript, wavPath)
	
	// Устанавливаем переменные окружения для Python
	env := os.Environ()
	env = append(env, "PYTHONIOENCODING=utf-8")
	env = append(env, "PYTHONUTF8=1")
	cmd.Env = env

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Запускаем процесс
	log.Printf("Начинаю распознавание речи из файла: %s", wavPath)
	err := cmd.Run()

	// Логируем вывод stderr в любом случае
	if stderrStr := stderr.String(); stderrStr != "" {
		log.Printf("Stderr: %s", stderrStr)
	}

	// Проверяем ошибки
	if err != nil {
		return "", fmt.Errorf("Ошибка при распознавании речи: %v", err)
	}

	// Возвращаем распознанный текст
	return stdout.String(), nil
}

func cleanup() {
	// Очищаем временные файлы
	files := []string{
		filepath.Join(config.Paths.TempDir, "voice.wav"),
		filepath.Join(config.Paths.TempDir, "voice.ogg"),
	}

	for _, file := range files {
		if err := os.Remove(file); err != nil && !os.IsNotExist(err) {
			log.Printf("Ошибка при удалении файла %s: %v", file, err)
		}
	}
}

func voiceMessageHandler(update tgbotapi.Update, bot *tgbotapi.BotAPI) {
	// Получаем информацию о пользователе
	username := update.Message.From.UserName
	if username == "" {
		username = update.Message.From.FirstName
	}

	// Проверяем длительность
	duration := update.Message.Voice.Duration
	if duration > config.Telegram.MaxVoiceDuration { 
		msg := tgbotapi.NewMessage(update.Message.Chat.ID, 
			"⚠️ <b>Слишком длинное голосовое сообщение</b>\n\n"+
			"Максимальная длительность: "+formatDuration(config.Telegram.MaxVoiceDuration)+"\n"+
			"Ваше сообщение: "+formatDuration(duration))
		msg.ParseMode = "HTML"
		bot.Send(msg)
		return
	}

	// Создаем начальное сообщение
	initialMessage := fmt.Sprintf(`🎤 <b>Распознаю голосовое сообщение</b>
👤 От: @%s
⏱ Длительность: %s

<i>Пожалуйста, подождите...</i>`, username, formatDuration(duration))

	msg := tgbotapi.NewMessage(update.Message.Chat.ID, initialMessage)
	msg.ParseMode = "HTML"
	processingMsg, err := bot.Send(msg)
	if err != nil {
		log.Printf("Ошибка при отправке сообщения: %v", err)
		return
	}

	// Создаем временную директорию, если её нет
	if err := os.MkdirAll(config.Paths.TempDir, os.ModePerm); err != nil {
		log.Printf("Ошибка при создании временной директории: %v", err)
		editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
			"❌ <b>Ошибка при создании временной директории</b>")
		return
	}

	// Получаем информацию о голосовом сообщении
	file, err := bot.GetFile(tgbotapi.FileConfig{FileID: update.Message.Voice.FileID})
	if err != nil {
		log.Printf("Ошибка при получении файла: %v", err)
		editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
			"❌ <b>Ошибка при получении голосового сообщения</b>")
		return
	}

	time.Sleep(300 * time.Millisecond)
	editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
		fmt.Sprintf(`🎤 <b>Распознаю голосовое сообщение</b>
👤 От: @%s
⏱ Длительность: %s

⏳ <i>Загружаю голосовое сообщение...</i>`, username, formatDuration(duration)))

	// Загружаем OGG файл
	oggPath := filepath.Join(config.Paths.TempDir, "voice.ogg")
	err = downloadFile(oggPath, file.Link(config.Telegram.Token))
	if err != nil {
		log.Printf("Ошибка при загрузке файла: %v", err)
		editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
			"❌ <b>Ошибка при загрузке голосового сообщения</b>")
		return
	}

	// Проверяем, что OGG файл существует и не пустой
	if info, err := os.Stat(oggPath); os.IsNotExist(err) {
		log.Printf("OGG файл не существует после загрузки: %v", err)
		editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
			"❌ <b>Ошибка при сохранении голосового сообщения</b>")
		return
	} else if info.Size() == 0 {
		log.Printf("Загружен пустой OGG файл")
		editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
			"❌ <b>Получен пустой файл</b>")
		os.Remove(oggPath)
		return
	}

	time.Sleep(300 * time.Millisecond)
	editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
		fmt.Sprintf(`🎤 <b>Распознаю голосовое сообщение</b>
👤 От: @%s
⏱ Длительность: %s

🔄 <i>Конвертирую аудио...</i>`, username, formatDuration(duration)))

	// Конвертируем в WAV
	wavPath := filepath.Join(config.Paths.TempDir, "voice.wav")
	err = convertOggToWav(oggPath, wavPath)
	if err != nil {
		log.Printf("Ошибка при конвертации файла: %v", err)
		editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
			"❌ <b>Ошибка при конвертации голосового сообщения</b>")
		os.Remove(oggPath)
		return
	}

	// Удаляем OGG файл, он больше не нужен
	os.Remove(oggPath)

	// Проверяем WAV файл перед распознаванием
	if info, err := os.Stat(wavPath); os.IsNotExist(err) {
		log.Printf("WAV файл не существует после конвертации: %v", err)
		editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
			"❌ <b>Ошибка при подготовке аудио</b>")
		return
	} else if info.Size() == 0 {
		log.Printf("WAV файл пустой после конвертации")
		editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
			"❌ <b>Ошибка при конвертации аудио</b>")
		os.Remove(wavPath)
		return
	}

	time.Sleep(300 * time.Millisecond)
	editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
		fmt.Sprintf(`🎤 <b>Распознаю голосовое сообщение</b>
👤 От: @%s
⏱ Длительность: %s

💫 <i>Загружаю модель Whisper (small)...</i>`, username, formatDuration(duration)))

	time.Sleep(300 * time.Millisecond)
	editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
		fmt.Sprintf(`🎤 <b>Распознаю голосовое сообщение</b>
👤 От: @%s
⏱ Длительность: %s

🔍 <i>Распознаю речь...</i>`, username, formatDuration(duration)))

	// Распознаем речь
	text, err := recognizeSpeech(wavPath)
	if err != nil {
		log.Printf("Ошибка при распознавании речи: %v", err)
		editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
			"❌ <b>Ошибка при распознавании речи</b>")
		os.Remove(wavPath)
		return
	}

	// Отправляем результат
	if text == "" {
		editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, 
			"❌ <b>Не удалось распознать текст</b>")
	} else {
		// Используем обычные переносы строк вместо HTML-тегов
		response := fmt.Sprintf(`✅ <b>Распознанный текст</b>
👤 От: @%s
⏱ Длительность: %s

%s`, username, formatDuration(duration), text)
		if err := editMessageHTML(bot, update.Message.Chat.ID, processingMsg.MessageID, response); err != nil {
			log.Printf("Ошибка при отправке результата: %v", err)
			// В случае ошибки пробуем отправить без форматирования
			plainResponse := fmt.Sprintf("✅ Распознанный текст\nОт: @%s\nДлительность: %s\n\n%s",
				username, formatDuration(duration), text)
			msg := tgbotapi.NewEditMessageText(update.Message.Chat.ID, processingMsg.MessageID, plainResponse)
			if _, err := bot.Send(msg); err != nil {
				log.Printf("Ошибка при отправке plain текста: %v", err)
			}
		}
	}

	// Удаляем WAV файл только после отправки сообщения
	os.Remove(wavPath)
}

func formatDuration(seconds int) string {
	if seconds < 60 {
		return fmt.Sprintf("%d сек", seconds)
	}
	minutes := seconds / 60
	remainingSeconds := seconds % 60
	if remainingSeconds == 0 {
		return fmt.Sprintf("%d мин", minutes)
	}
	return fmt.Sprintf("%d мин %d сек", minutes, remainingSeconds)
}

func editMessageHTML(bot *tgbotapi.BotAPI, chatID int64, messageID int, text string) error {
	msg := tgbotapi.NewEditMessageText(chatID, messageID, text)
	msg.ParseMode = "HTML"
	_, err := bot.Send(msg)
	return err
}

func handleStart(bot *tgbotapi.BotAPI, update tgbotapi.Update) error {
	welcomeText := `👋 Привет! Я бот для распознавания голосовых сообщений.

🎙 Отправь мне голосовое сообщение, и я преобразую его в текст.

🌍 Поддерживаемые языки:
`
	// Добавляем список языков из конфигурации
	for code, name := range config.Speech.Languages {
		welcomeText += fmt.Sprintf("• %s (%s)\n", name, code)
	}

	welcomeText += `
⚙️ По умолчанию используется русский язык.
🔄 Автоматическое определение языка: `

	if config.Speech.AutoDetectLanguage {
		welcomeText += "включено"
	} else {
		welcomeText += "выключено"
	}

	msg := tgbotapi.NewMessage(update.Message.Chat.ID, welcomeText)
	_, err := bot.Send(msg)
	return err
}

func handleHelp(bot *tgbotapi.BotAPI, update tgbotapi.Update) error {
	helpText := `🤖 VoicePidor Bot - Помощь

Основные команды:
/start - Начать работу с ботом
/help - Показать это сообщение

🎙 Как использовать:
1. Отправьте голосовое сообщение
2. Дождитесь обработки
3. Получите текст

🌍 Поддерживаемые языки:
`
	// Добавляем список языков из конфигурации
	for code, name := range config.Speech.Languages {
		helpText += fmt.Sprintf("• %s (%s)\n", name, code)
	}

	helpText += fmt.Sprintf(`
⚙️ Текущие настройки:
• Язык по умолчанию: %s
• Автоопределение языка: %v
• Модель Whisper: %s
• GPU: %v

ℹ️ Максимальная длительность: %d секунд`, 
		config.Speech.DefaultLanguage,
		config.Speech.AutoDetectLanguage,
		config.Speech.Model,
		config.Speech.UseGPU,
		config.Telegram.MaxVoiceDuration)

	msg := tgbotapi.NewMessage(update.Message.Chat.ID, helpText)
	_, err := bot.Send(msg)
	return err
}

func main() {
	cleanup()

	bot, err := tgbotapi.NewBotAPI(config.Telegram.Token)
	if err != nil {
		log.Panic(err)
	}

	bot.Debug = false

	deleteWebhook := tgbotapi.DeleteWebhookConfig{
		DropPendingUpdates: true,
	}
	_, err = bot.Request(deleteWebhook)
	if err != nil {
		log.Printf("Ошибка при удалении webhook: %v", err)
	}

	log.Printf("Бот успешно запущен, ID: %d, Имя: %s", bot.Self.ID, bot.Self.UserName)

	updateConfig := tgbotapi.NewUpdate(0)
	updateConfig.Timeout = 60

	updates := bot.GetUpdatesChan(updateConfig)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		for update := range updates {
			if update.Message == nil {
				continue
			}

			log.Printf("Получено обновление ID: %d", update.UpdateID)

			switch {
			case update.Message.Command() == "start":
				if err := handleStart(bot, update); err != nil {
					log.Printf("Ошибка при обработке команды /start: %v", err)
				}
			case update.Message.Command() == "help":
				if err := handleHelp(bot, update); err != nil {
					log.Printf("Ошибка при обработке команды /help: %v", err)
				}
			case update.Message.Voice != nil:
				voiceMessageHandler(update, bot)
			}
		}
	}()

	<-sigChan
	log.Println("Получен сигнал завершения, закрываю бота...")
	cleanup()
}
