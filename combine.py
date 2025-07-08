import threading
import queue
import time
import requests
from flask import Flask, request
from gtts import gTTS
import os
import logging

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# -----------------------------
# 🔈 Flask 음성 피드백 서버
# -----------------------------
app = Flask(__name__)


def speak(text):
    tts = gTTS(text=text, lang="ko")
    tts.save("speech.mp3")
    os.system("mpg123 -q speech.mp3")


@app.route("/notify", methods=["POST"])
def notify():
    data = request.get_json()
    cmd = data.get("cmd")
    if cmd == "light_on":
        speak("조명을 켭니다")
    elif cmd == "light_off":
        speak("조명을 끕니다")
    elif cmd == "motor_on":
        speak("선풍기를 켭니다")
    elif cmd == "motor_off":
        speak("선풍기를 끕니다")
    else:
        print("🔇 명령 없음:", cmd)
    return "OK", 200


def run_flask_server():
    print("🚀 Flask 음성 서버 시작 (port 8000)")
    app.run(host="0.0.0.0", port=8000, debug=False)


# -----------------------------
# 🧠 LLM & Queue
# -----------------------------
import json


def ask_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:1b",
            "prompt": f"""
...너의 프롬프트 그대로...
문장: "{prompt}"
정답:""",
            "stream": False,
        },
    )
    return response.json()["response"].strip()


def send_worker():
    while True:
        cmd = cmd_queue.get()
        try:
            # Raspberry Pi 전송
            res1 = requests.post("http://10.10.15.195:5000/control", json={"cmd": cmd})
            print("📡 Pi 응답:", res1.text)
        except Exception as e:
            print("❌ Pi 전송 실패:", e)

        try:
            # 스피커 전송
            res2 = requests.post("http://localhost:8000/notify", json={"cmd": cmd})
            print("🔊 스피커 응답:", res2.text)
        except Exception as e:
            print("❌ 스피커 전송 실패:", e)
        cmd_queue.task_done()


# -----------------------------
# 🧵 메인
# -----------------------------
if __name__ == "__main__":
    cmd_queue = queue.Queue()

    # Flask 서버 + Worker 스레드 시작
    threading.Thread(target=run_flask_server, daemon=True).start()
    threading.Thread(target=send_worker, daemon=True).start()
    time.sleep(1)

    while True:
        user_input = input("명령 입력 (exit 입력 시 종료): ")
        if user_input.lower() == "exit":
            break

        cmd = ask_ollama(user_input)
        print("📥 LLM 응답:", cmd)

        if cmd in ["light_on", "light_off", "motor_on", "motor_off"]:
            cmd_queue.put(cmd)
        else:
            print("⚠️ 실행할 명령이 아님 (unknown)")
