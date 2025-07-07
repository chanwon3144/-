import requests

def ask_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:1b",
            "prompt": f"""
너는 smart home을 제어하는 모델이야 다음 문장을 반드시 아래 중 하나로만 변환해: light_on / light_off / motor_on / motor_off
예시:
- "불 켜" → light_on
- "조명 켜줘" → light_on
- "불 꺼줘" → light_off
- "조명 꺼" → light_off
- "light on" → light_on
- "light off" → light_off
- "선풍기 켜줘" → motor_on
- "선풍기 꺼줘" → motor_off 
- "fan on" → motor_on
- "fan off" → motor_off

문장: "{prompt}"
정답:""",
            "stream": False
        }
    )
    return response.json()["response"].strip()

def send_to_raspberry(cmd):
    pi_url = "http://10.10.15.195:5000/control"  # ← 여기 IP 수정!
    try:
        res = requests.post(pi_url, json={"cmd": cmd})
        print("📡 Pi 응답:", res.text)
    except Exception as e:
        print("❌ 전송 실패:", e)

# 메인 루프
while True:
    user_input = input("명령 입력 (exit 입력 시 종료): ")
    if user_input.lower() == "exit":
        break

    cmd = ask_ollama(user_input)
    print("📥 LLM 응답:", cmd)

    if cmd in ["light_on", "light_off"]:
        send_to_raspberry(cmd)
    elif cmd in ["motor_on", "motor_off"]:
        send_to_raspberry(cmd)
    else:
        print("⚠️ 실행할 명령이 아님 (unknown)")
