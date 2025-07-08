import threading
from flask import Flask, request
from gtts import gTTS
import os
import time
import requests
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # 또는 logging.CRITICAL
import subprocess

# -----------------------------
# 🔈 Flask 음성 피드백 서버
# -----------------------------
app = Flask(__name__)

def speak(text):
    tts = gTTS(text=text, lang='ko')
    tts.save("speech.mp3")
    subprocess.run(["mpg123", "-q"," speech.mp3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
# 🤖 자연어 명령 처리 루프
# -----------------------------
def ask_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:1b",
            "prompt": f"""
너는 smart home을 제어하는 모델이야 다음 문장을 반드시 아래 중 하나로만 변환해. 근데 문장은 직접 명령일 수도 있고 감정/상태 표현일 수도 있어: light_on / light_off / motor_on / motor_off
문장: "{prompt}"
정답:""",
            "stream": False
        }
    )
    return response.json()["response"].strip()

def send_to_raspberry(cmd):
    pi_url = "http://10.10.15.195:5000/control"  # 🖐 IP 주소 확인!
    try:
        res = requests.post(pi_url, json={"cmd": cmd})
        print("📡 Pi 응답:", res.text)
    except Exception as e:
        print("❌ Pi 전송 실패:", e)

def send_to_speaker(cmd):
    try:
        res = requests.post("http://localhost:8000/notify", json={"cmd": cmd})
        print("🔊 스피커 응답:", res.text)
    except Exception as e:
        print("❌ 스피커 전송 실패:", e)

# -----------------------------
# 🧠 메인 실행부
# -----------------------------
if __name__ == "__main__":
    # Flask 서버 백그라운드 실행
    threading.Thread(target=run_flask_server, daemon=True).start()
    time.sleep(1)  # 서버 부팅 대기

    while True:
        user_input = input("명령 입력 (exit 입력 시 종료): ")
        if user_input.lower() == "exit":
            break

        cmd = ask_ollama(user_input)
        print("📥 LLM 응답:", cmd)

        if cmd in ["light_on", "light_off", "motor_on", "motor_off"]:
            send_to_raspberry(cmd)
            send_to_speaker(cmd)
        else:
            print("⚠️ 실행할 명령이 아님 (unknown)")

