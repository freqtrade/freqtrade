import base64
import os
from pathlib import Path


def mp3_to_base64(mp3_path):
    with open(mp3_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def create_html(audio_data_list, output_file="audio_player.html"):
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>MP3 Audio Player</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .audio-container {
            margin-bottom: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        audio {
            width: 100%;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>Аудио Плеер</h1>
    {audio_players}
</body>
</html>
"""

    audio_players = ""
    for i, (filename, audio_data) in enumerate(audio_data_list, 1):
        audio_players += f"""
        <div class="audio-container">
            <h3>Аудио {i}: {filename}</h3>
            <audio controls>
                <source src="data:audio/mp3;base64,{audio_data}" type="audio/mp3">
                Ваш браузер не поддерживает аудио элемент.
            </audio>
        </div>
        """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template.format(audio_players=audio_players))


def main():
    # Получаем список MP3 файлов в текущей директории
    mp3_files = list(Path(".").glob("*.mp3"))

    if not mp3_files:
        print("В текущей директории не найдено MP3 файлов.")
        return

    print(f"Найдено {len(mp3_files)} MP3 файлов:")
    for i, file in enumerate(mp3_files, 1):
        print(f"{i}. {file.name}")

    # Конвертируем каждый MP3 в base64
    audio_data_list = []
    for mp3_file in mp3_files:
        try:
            print(f"Обработка файла: {mp3_file.name}...")
            audio_data = mp3_to_base64(mp3_file)
            audio_data_list.append((mp3_file.name, audio_data))
        except Exception as e:
            print(f"Ошибка при обработке файла {mp3_file.name}: {e}")

    if audio_data_list:
        output_file = "audio_player.html"
        create_html(audio_data_list, output_file)
        print(f"\nГотово! HTML файл сохранен как: {output_file}")
        print(f"Откройте {output_file} в браузере для прослушивания аудио.")
    else:
        print("Не удалось обработать ни одного MP3 файла.")


if __name__ == "__main__":
    main()
