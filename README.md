# IoT 기반 제스처 스마트홈 케어 시스템

![Project Logo or Diagram Placeholder](https://via.placeholder.com/600x300?text=Gesture+Smart+Home+System)
_프로젝트를 나타내는 로고나 시스템 구성도 이미지를 여기에 추가하면 좋습니다._

## 💡 프로젝트 소개 (Introduction)

본 프로젝트는 **IoT(사물 인터넷) 기반의 제스처 스마트홈 케어 시스템**입니다. 사용자의 손 제스처를 인식하여 집 안의 스마트 기기를 직관적으로 제어할 수 있도록 돕습니다. 특히 신체가 불편한 노약자나 장애인 등 특정 사용자층에게 더욱 편리하고 자연스러운 스마트홈 환경을 제공하는 것을 목표로 합니다.

## ✨ 주요 기능 (Features)

* **정교한 손 제스처 인식:** **직접 수집하고 학습시킨 2000장 이상의 손동작 데이터셋을 활용하여** MediaPipe 기반의 손 감지 및 제스처 인식의 정확도를 높였습니다. 웹캠으로 사용자의 손을 실시간으로 감지하고 다양한 제스처(예: 손 펴기, 주먹 쥐기)를 인식합니다.
* **LLM 기반 의도 파악:** 인식된 제스처 정보를 경량 LLM(대규모 언어 모델)인 Ollama의 **Mistral 모델**에 전달하여 사용자의 의도(예: 조명 켜기, 에어컨 끄기)를 파악하고 명령을 생성합니다.
* **라즈베리 파이 GPIO 제어:** LLM이 생성한 명령에 따라 라즈베리 파이의 GPIO(General Purpose Input/Output) 핀을 제어하여 실제 물리적인 스마트 기기(예: LED, 버저, 모터 등)를 작동시킵니다.
* **직관적인 음성 피드백:** TTS(Text-to-Speech)를 통해 시스템의 상태나 LLM의 응답을 음성으로 사용자에게 알려주어 사용 편의성을 높입니다.

## 🛠️ 기술 스택 (Technologies)

* **Language:** Python 3
* **Computer Vision:** MediaPipe, OpenCV (`cv2`)
* **Hardware Interface:** `RPi.GPIO`
* **Large Language Model (LLM):** Ollama (Local LLM Server), **Mistral**
* **Text-to-Speech (TTS):** `gTTS`
* **Communication:** `requests` (Ollama API 통신)
* **Hardware:** Raspberry Pi (권장: 4 또는 5), USB 웹캠, 스피커, LED (또는 제어할 기타 스마트 기기)

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

## 🏃‍♂️ 실행 방법 (How to Run)

가상 환경이 활성화된 터미널에서 다음 명령어를 실행합니다. **GPIO 제어를 위해 `sudo` 권한이 필수적이며, 가상 환경의 파이썬 인터프리터를 명시적으로 지정해야 합니다.**

```bash
# 가상 환경이 활성화된 상태에서 (프롬프트에 (mediapipe_env) 확인)
sudo /home/chanwon/mediapipe_env/bin/python3 gesture_controller.py
```

```
# 로컬 터미널에서 LLm 모델 ollama mistral을 실행합니다.
python3 start_command.py
```
