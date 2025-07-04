from flask import Flask, request, jsonify
import RPi.GPIO as GPIO
import time

app = Flask(__name__)

# --- GPIO 핀 설정 ---
LIGHT_RELAY_PIN = 17
MOTOR_IN1_PIN = 23
MOTOR_IN2_PIN = 24
MOTOR_ENA_PIN = 18

# GPIO 초기화
GPIO.setmode(GPIO.BCM)

# 조명 핀 설정
GPIO.setup(LIGHT_RELAY_PIN, GPIO.OUT)
GPIO.output(LIGHT_RELAY_PIN, GPIO.LOW) # 초기 상태: 조명 OFF

# 모터 핀 설정
GPIO.setup(MOTOR_IN1_PIN, GPIO.OUT)
GPIO.setup(MOTOR_IN2_PIN, GPIO.OUT)
GPIO.setup(MOTOR_ENA_PIN, GPIO.OUT)

# PWM 객체 생성 (모터용, 주파수 100Hz)
pwm_motor = GPIO.PWM(MOTOR_ENA_PIN, 100)
pwm_motor.start(0) # 초기 듀티 사이클 0 (모터 정지)

# --- 모터 제어 함수 ---
def set_motor(direction, speed_percentage):
    speed_percentage = max(0, min(100, speed_percentage))

    if direction == "forward":
        GPIO.output(MOTOR_IN1_PIN, GPIO.HIGH)
        GPIO.output(MOTOR_IN2_PIN, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(MOTOR_IN1_PIN, GPIO.LOW)
        GPIO.output(MOTOR_IN2_PIN, GPIO.HIGH)
    else: # "stop"
        GPIO.output(MOTOR_IN1_PIN, GPIO.LOW)
        GPIO.output(MOTOR_IN2_PIN, GPIO.LOW)
        
    pwm_motor.ChangeDutyCycle(speed_percentage)
    print(f"⚙️ 모터 {direction} 방향, {speed_percentage}% 속도 설정.")

# --- 통합 제어 엔드포인트 ---
@app.route('/control', methods=['POST'])
def control_device():
    data = request.get_json()
    cmd = data.get("cmd")

    if cmd == "light_on":
        GPIO.output(LIGHT_RELAY_PIN, GPIO.HIGH)
        print("💡 조명 ON")
        return jsonify({"status": "light_on"}), 200
    elif cmd == "light_off":
        GPIO.output(LIGHT_RELAY_PIN, GPIO.LOW)
        print("💡 조명 OFF")
        return jsonify({"status": "light_off"}), 200
    elif cmd == "motor_on":
        set_motor("forward", 70) # fan_on 제스처에 해당
        return jsonify({"status": "motor_on", "direction": "forward", "speed": 70}), 200
    elif cmd == "motor_off":
        set_motor("stop", 0) # fan_off 제스처에 해당
        return jsonify({"status": "motor_off"}), 200
    else:
        print("❓ 알 수 없는 명령:", cmd)
        return jsonify({"error": "unknown command"}), 400

@app.route('/')
def health_check():
    return "✅ 통합 GPIO 제어 서버 실행 중!", 200

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        pwm_motor.stop()
        GPIO.cleanup()
        print("GPIO 정리 완료.")
