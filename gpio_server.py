# gpio_server.py
from flask import Flask, request, jsonify
import RPi.GPIO as GPIO

app = Flask(__name__)

RELAY_PIN = 17  # 릴레이 연결된 GPIO 핀 번호 (필요 시 변경)

# GPIO 초기화
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)

@app.route('/light', methods=['POST'])
def control_light():
    data = request.get_json()
    cmd = data.get("cmd")

    if cmd == "light_on":
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        print("💡 조명 ON")
        return jsonify({"status": "on"}), 200

    elif cmd == "light_off":
        GPIO.output(RELAY_PIN, GPIO.LOW)
        print("💡 조명 OFF")
        return jsonify({"status": "off"}), 200

    else:
        print("❓ 알 수 없는 명령:", cmd)
        return jsonify({"error": "unknown command"}), 400

@app.route('/')
def health_check():
    return "✅ GPIO 제어 서버 실행 중!", 200

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        GPIO.cleanup()

