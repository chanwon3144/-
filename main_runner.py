import subprocess
import time
import os

# 각 스크립트 파일명 (실제 경로에 맞게 수정 필요)
# 예를 들어, 모든 파일이 같은 디렉토리에 있다면 파일명만 적어도 됩니다.
SCRIPTS = [
    "hand_changed_h5.py",    # 제스처 인식 코드 (이름이 gesture_recognition.py로 바뀌었으면 그 이름으로)
    "human_detect_buzzer.py", # 사람 인식 코드
    "local_server.py",       # LLM 채팅 서버 (이름이 local_server.py가 맞다면)
    "gpio_server.py"         # GPIO 제어 서버
]

# 실행된 프로세스들을 저장할 리스트
processes = []

print("모든 프로젝트 스크립트를 시작합니다...")

try:
    for script in SCRIPTS:
        print(f"✅ {script} 시작 중...")
        # 'python3' 대신 'python'을 사용하거나, 환경에 맞게 조정
        # '-u' 옵션은 버퍼링 없이 즉시 표준 출력을 표시하게 합니다 (디버깅에 유용).
        process = subprocess.Popen(['python3', '-u', script])
        processes.append(process)
        time.sleep(1) # 각 스크립트가 시작될 시간을 약간 줍니다.

    print("\n🚀 모든 스크립트가 성공적으로 시작되었습니다!")
    print("종료하려면 Ctrl+C를 누르세요.")

    # 모든 서브 프로세스가 종료될 때까지 기다리거나, 메인 스크립트를 계속 실행
    # 여기서는 Ctrl+C로 종료할 때까지 대기하도록 함
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nCtrl+C가 감지되었습니다. 모든 스크립트를 종료합니다...")
    for process in processes:
        if process.poll() is None: # 아직 실행 중인 프로세스인 경우
            process.terminate() # 프로세스에 종료 신호를 보냄 (Graceful termination)
            print(f"❌ 프로세스 {process.args[2]} 종료 신호 전송.")
    
    # 프로세스들이 완전히 종료될 때까지 기다림 (선택 사항)
    for process in processes:
        try:
            process.wait(timeout=5) # 최대 5초 대기
        except subprocess.TimeoutExpired:
            print(f"⚠️ 프로세스 {process.args[2]}가 응답하지 않아 강제 종료합니다.")
            process.kill() # 강제 종료
            
finally:
    print("모든 스크립트 종료 및 정리 완료.")