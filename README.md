# IoT 기반 제스처 스마트홈 케어 시스템

![Project Logo or Diagram Placeholder](https://via.placeholder.com/600x300?text=Gesture+Smart+Home+System)
_프로젝트를 나타내는 로고나 시스템 구성도 이미지를 여기에 추가하면 좋습니다._

## 💡 프로젝트 소개 (Introduction)

본 프로젝트는 **IoT(사물 인터넷) 기반의 제스처 스마트홈 케어 시스템**입니다. 사용자의 손 제스처를 인식하여 집 안의 스마트 기기를 직관적으로 제어할 수 있도록 돕습니다. 특히 신체가 불편한 노약자나 장애인 등 특정 사용자층에게 더욱 편리하고 자연스러운 스마트홈 환경을 제공하는 것을 목표로 합니다.

## ✨ 주요 기능

- ✋ **정교한 손 제스처 인식**  
  > MediaPipe 기반 실시간 손 인식 + 자체 수집한 2000장 이상의 데이터셋

- 🧠 **LLM 기반 사용자 의도 파악**  
  > Ollama 로컬 서버에 구동된 Gemma 3 1B 모델을 통해 사용자의 의도 파악

- 💡 **GPIO를 통한 스마트 기기 제어**  
  > 제스처를 통해 LED, 모터 등의 물리 기기 제어

- 🧍 **YOLOv5 기반 사람 감지 및 부저 알림**  
  > 사람 감지 시 자동으로 부저 울림

- 🔊 **TTS 기반 음성 피드백**  
  > 시스템 응답을 음성으로 출력하여 사용 편의성 향상
  
## 🛠️ 기술 스택

| 분류 | 사용 기술 |
|------|-----------|
| Language | Python 3 |
| Computer Vision | MediaPipe, OpenCV, YOLOv5 |
| AI/LLM | Ollama (Gemma 3 1B) |
| TTS | gTTS |
| GPIO 제어 | RPi.GPIO |
| 하드웨어 | Raspberry Pi (4 이상), 웹캠, 스피커, LED, 부저, 모터 등 |


## 🚀 시작하기 (Getting Started)

### 📝 사전 준비물 (Prerequisites)

* **물리적인 라즈베리 파이 보드 (필수!)**: 라즈베리 파이 3B+ 이상 권장
* Raspberry Pi OS (64-bit Lite 또는 Desktop) 설치
* 웹캠 (라즈베리 파이와 호환되는 USB 웹캠)
* 스피커 (TTS 음성 출력을 위해 필요)
* 인터넷 연결 (Ollama 모델 다운로드 및 `gTTS` 사용 시 필요)
* **Ollama 서버 설치 및 Mistral 모델 다운로드:**
    ```bash
    curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
    ollama run mistral # mistral 모델 다운로드 및 실행 (최초 1회)
    ```
    _`ollama` 서버는 스크립트 실행 전에 백그라운드에서 실행 중이어야 합니다._

### 📦 설치 방법 (Installation)

1.  **프로젝트 클론:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git) # 본인의 깃허브 경로로 변경
    cd your-repo-name # 프로젝트 폴더로 이동
    ```

2.  **파이썬 가상 환경 생성 및 활성화:**
    ```bash
    python3 -m venv mediapipe_env
    source mediapipe_env/bin/activate
    ```

3.  **필요한 라이브러리 설치:**
    ```bash
    pip install opencv-python-headless mediapipe RPi.GPIO requests gtts
    ```
    _(`opencv-python-headless` 대신 `opencv-python`을 사용할 수도 있습니다.)_

### 🔌 하드웨어 연결 (Hardware Setup)

* **LED 연결:** 예시 코드는 GPIO 17번 핀(BCM 모드)을 LED 제어에 사용합니다.
    * 라즈베리 파이 **GPIO 17번 핀**과 LED의 **양극(+)**을 연결합니다.
    * LED의 **음극(-)**을 적절한 **저항(예: 220옴~330옴)**을 통해 라즈베리 파이의 **GND(접지) 핀**에 연결합니다.
    * _자세한 내용은 라즈베리 파이 GPIO 핀아웃 다이어그램을 참고하세요._
![image](https://github.com/user-attachments/assets/e7d6eb58-bc17-43c6-81a1-95f3065a03ce)


## 🏃‍♂️ 실행 방법 (How to Run) /[detect.py, gpio_server.py, start_command.py]

가상 환경이 활성화된 터미널에서 다음 명령어를 실행합니다. **GPIO 제어를 위해 `sudo` 권한이 필수적이며, 가상 환경의 파이썬 인터프리터를 명시적으로 지정해야 합니다.**

```bash
# 가상 환경이 활성화된 상태에서 (프롬프트에 (mediapipe_env) 확인)
sudo /home/chanwon/mediapipe_env/bin/python3 gesture_controller.py
```

```
# 로컬 터미널에서 LLm 모델 ollama mistral을 실행합니다.
python3 start_command.py

# 카메라 ON
python3 detect.py --weights best.pt --source 0 --conf 0.4 --img 416 --device cpu

# HTTP 요청으로 간접 제어를 위한 Flask 서버 띄우기
python3 gpio_server.py
```
## 🏃‍♂️ 실행 방법 (project_ai)

본 프로젝트는 2개의 터미널을 사용해 실행됩니다.
🖥️ 터미널 1 – GPIO 서버 실행

    프로젝트 디렉토리로 이동:

cd /home/chanwon/YOLO/yolov5/project_ai/

GPIO 서버 실행 (루트 권한 필요):

    sudo python3 gpio_server.py

        실행 후 Running on http://0.0.0.0:5000 메시지가 출력되면 성공입니다.

        ⚠️ 이 터미널은 종료하지 말고 계속 열어두세요.

🖥️ 터미널 2 – 제스처 인식 및 제어 실행

    같은 디렉토리로 이동:

cd /home/chanwon/YOLO/yolov5/project_ai/

가상 환경 활성화:

source .env/bin/activate

    프롬프트에 (.env) 표시가 나타나면 성공입니다.

제스처 인식 코드 실행:

python3 main.py

    웹캠 화면이 나타나고 실시간 제스처 인식이 시작됩니다.
