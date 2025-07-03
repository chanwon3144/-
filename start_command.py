import requests

def ask_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": f"""
다음 문장을 light_on / light_off / unknown 중 하나로만 변환해줘.
- "불 켜" → light_on
- "조명 켜줘" → light_on
- "불 꺼줘" → light_off
- "조명 꺼" → light_off
- "turn the light on" → light_on
- "light off please" → light_off
- "선풍기 꺼" → unknown
- "에어컨 켜" → unknown

문장: "{prompt}"
정답:""",
            "stream": False
        }
    )
    return response.json()["response"].strip()

def send_to_raspberry(cmd):
    pi_url = "http://10.10.15.195:5000/light"  # ← 여기 IP 수정!
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
    else:
        print("⚠️ 실행할 명령이 아님 (unknown)")

