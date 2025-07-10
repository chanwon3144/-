# IoT 기반 제스처 스마트홈 케어 시스템

[![Project Status: Active](https://img.shields.io/badge/status-active-success.svg)](https://github.com/your-username/your-repo-name)
[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PPT:** [https://www.canva.com/design/DAGsKYK5d9g/srHhvtxA6RWmiJ9MKy7bpw/edit?utm_content=DAGsKYK5d9g&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton](https://www.canva.com/design/DAGsKYK5d9g/srHhvtxA6RWmiJ9MKy7bpw/edit?utm_content=DAGsKYK5d9g&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

초안: https://www.canva.com/design/DAGsp-QF_tU/j4NEllfYvNJRnHNS1coWew/edit?utm_content=DAGsp-QF_tU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

---

## 💡 프로젝트 소개 (Introduction)

본 프로젝트는 **라즈베리파이 기반의 스마트 홈케어 IoT 시스템 구축**을 목표로 합니다. MobileNetV2-Keras CNN을 활용한 손동작 및 사람 인식 기술과 LLM(Gemma3 1B) 기반의 자연어 처리 기능을 통합합니다. 이를 통해 사용자는 두 가지 직관적인 입력 방식을 모두 사용하여 LED 및 모터와 같은 기기를 실시간으로 제어할 수 있습니다. 시스템은 분산 아키텍처로, 라즈베리파이와 Ubuntu PC 간 통신을 통해 구성됩니다.

---

## ✨ 주요 기능 (Key Features)

- ✋ **정교한 손 제스처 인식**
  > MediaPipe 기반 실시간 손 인식 및 **자체 수집한 2000장 이상의 데이터셋**으로 학습된 모델을 활용합니다.

- 🧠 **LLM 기반 사용자 의도 파악**
  > **Ollama 로컬 서버에 구동된 Gemma 3 1B 모델**을 통해 사용자의 자연어 명령을 분석하고 의도를 파악합니다.

- 💡 **GPIO를 통한 스마트 기기 제어**
  > 인식된 제스처 또는 자연어 명령에 따라 라즈베리파이 GPIO 핀을 통해 LED, 모터 등 물리적 기기를 실시간으로 제어합니다. **모든 GPIO 제어는 `RPi.GPIO` 라이브러리를 사용합니다.**

- 🧍 **YOLOv5 기반 사람 감지 및 부저 알림**
  > 카메라를 통해 사람을 감지하면 자동으로 부저 알림이 작동하여 특정 상황을 사용자에게 즉시 알립니다.

- 🔊 **TTS 기반 음성 피드백**
  > **gTTS(Google Text-to-Speech) 라이브러리를 활용**하여 시스템의 제어 응답 및 상태 변화를 음성으로 출력, 사용 편의성과 직관성을 향상시킵니다.

---

## 🛠️ 기술 스택 (Tech Stack)

| 분류 | 사용 기술 |
|---|---|
| Language | Python 3 |
| Computer Vision | MediaPipe, OpenCV, YOLOv5, Keras (MobileNetV2) |
| AI/LLM | Ollama (Gemma 3 1B) |
| TTS | **gTTS**, mpg123 |
| GPIO 제어 | **RPi.GPIO** |
| 웹 프레임워크 | Flask, Requests |
| 하드웨어 | Raspberry Pi (4 이상), 웹캠, 스피커, LED, 부저, 모터, 점프선, 저항 |

---

## 🚀 시작하기 (Getting Started)

### 📝 사전 준비물 (Prerequisites)

* **하드웨어:**
    * **라즈베리 파이 보드 (필수)**: Raspberry Pi 3B+ 이상 (최신 버전 권장, 특히 Pi 4 이상)
    * Raspberry Pi OS (64-bit Lite 또는 Desktop) 설치 완료
    * **Ubuntu가 설치된 별도 PC (필수)**: LLM 및 TTS 서버 구동용
    * USB 웹캠 (라즈베리 파이와 호환)
    * 스피커 (3.5mm 잭 또는 USB 스피커, 음성 피드백 출력용 - Ubuntu PC에 연결)
    * LED, 부저, 모터 (제어할 실제 기기 - 라즈베리파이에 연결)
    * 점퍼선, 저항 등 기본적인 전자 부품
    * 안정적인 네트워크 연결 (라즈베리파이와 Ubuntu PC 간 통신용)

* **소프트웨어 (Ollama 서버 - Ubuntu PC에 설치):**
    * Ollama 서버가 Ubuntu PC에 설치되어 실행 중이어야 합니다.
    * `Ollama` 설치:
        ```bash
        curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
        ```
    * **Gemma 3 1B 모델 다운로드 및 실행 (최초 1회):**
        ```bash
        ollama run gemma:2b # gemma 2b 모델은 gemma 3 1B와 유사하며, 라즈베리파이에서 더 안정적일 수 있습니다. 필요시 gemma:latest (기본 7b) 또는 다른 모델을 선택하세요.
        ```
        _`ollama` 서버는 `local_sever.py` 실행 전에 **백그라운드에서 항상 실행 중**이어야 합니다._

### 📦 설치 방법 (Installation)

1.  **프로젝트 클론 (각 장치에서 실행):**
    * **라즈베리파이와 Ubuntu PC 모두에서 프로젝트를 클론합니다.**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git) # 본인의 GitHub 저장소 URL로 변경해주세요.
    cd your-repo-name # 프로젝트 폴더로 이동 (예: cd project_ai)
    ```

2.  **필요한 파이썬 라이브러리 설치:**
    * **라즈베리파이에서 설치:**
        ```bash
        pip install opencv-python-headless mediapipe tensorflow-cpu RPi.GPIO requests ultralytics
        ```
    * **Ubuntu PC에서 설치:**
        ```bash
        pip install opencv-python-headless requests gtts flask ultralytics
        ```
        * `tensorflow-cpu`는 라즈베리파이용 Keras 모델 실행을 위함입니다.
        * `opencv-python-headless`는 GUI 없이 OpenCV를 사용 시 권장됩니다 (데스크톱 환경에서는 `opencv-python`).
        * `ultralytics`는 YOLOv5 모델 사용을 위해 필요합니다.
        * **`gtts`는 TTS 기능 구현을 위해 필수입니다.**

3.  **추가 시스템 패키지 설치:**
    음성 재생을 위해 `mpg123`를 설치합니다.
    * **Ubuntu PC에서 설치:**
        ```bash
        sudo apt update
        sudo apt install mpg123
        ```
    * _라즈베리파이에서는 `mpg123`가 필요하지 않습니다, 음성 출력은 Ubuntu PC에서 담당합니다._

### 🔌 하드웨어 연결 (Hardware Setup)

* **LED 연결:** 예시 코드는 GPIO **17번 핀 (BCM 모드)**을 LED 제어에 사용합니다.
    * 라즈베리 파이 **GPIO 17번 핀**과 LED의 **양극(+)**을 연결합니다.
    * LED의 **음극(-)**을 적절한 **저항(예: 220옴~330옴)**을 통해 라즈베리 파이의 **GND(접지) 핀**에 연결합니다.
    * _**주의:** 저항 없이 LED를 직접 연결하면 LED가 손상될 수 있습니다._
* **부저 연결:** 부저는 GPIO **13번 핀 (BCM 모드)**을 사용합니다.
    * 라즈베리 파이 **GPIO 13번 핀**과 부저의 **신호 핀**을 연결합니다.
    * 부저의 **GND 핀**을 라즈베리 파이의 **GND 핀**에 연결합니다.
* **모터 연결:** 모터는 GPIO 핀에 직접 연결하기보다 모터 드라이버(예: L298N)를 통해 연결하는 것이 일반적입니다. 코드에서 사용하는 핀 번호(예: GPIO 18, 23, 24)에 맞게 모터 드라이버를 라즈베리파이에 연결하고, 모터를 드라이버에 연결합니다.
* **웹캠 연결:** 웹캠은 라즈베리파이에 연결합니다.
* **스피커 연결:** 스피커는 Ubuntu PC에 연결합니다.

![GPIO Pinout Diagram](uploaded:image.png-0b13a96f-ef25-4d96-af6e-b12da627d23c)
_라즈베리 파이 GPIO 핀아웃 다이어그램을 참고하여 정확하게 연결하세요._

---

## 🏃‍♂️ 실행 방법 (How to Run)

본 프로젝트는 **라즈베리파이**와 **Ubuntu PC** 두 장치에서 각각 실행해야 하는 스크립트들로 구성됩니다. 각 스크립트가 서로 통신할 수 있도록 **IP 주소를 정확히 설정하는 것이 중요합니다.**

### 🌐 IP 주소 확인 및 설정

1.  **라즈베리파이의 IP 주소 확인:**
    라즈베리파이 터미널에서 다음 명령어를 실행하여 IP 주소를 확인합니다.
    ```bash
    hostname -I
    ```
    예시: `192.168.1.100`

2.  **Ubuntu PC의 IP 주소 확인:**
    Ubuntu 터미널에서 다음 명령어를 실행하여 IP 주소를 확인합니다.
    ```bash
    hostname -I
    ```
    예시: `192.168.1.101`

3.  **코드 내 IP 주소 업데이트:**
    확인된 IP 주소를 다음 파일들에 반영하여 수정합니다.
    * **`local_sever.py` (Ubuntu PC에서 실행될 파일):**
        `send_to_raspberry` 함수 내의 `pi_url` 변수를 라즈베리파이의 실제 IP 주소로 변경합니다.
        ```python
        # 변경 전: pi_url = "http://[Raspberry_Pi_IP_Address]:5000/control"
        pi_url = "[http://192.168.1.100:5000/control](http://192.168.1.100:5000/control)" # 예시
        ```
    * **`gesture_debounce_success.py` (라즈베리파이에서 실행될 파일):**
        `send_to_speaker` 함수 내의 `speaker_url` 변수를 Ubuntu PC의 실제 IP 주소로 변경합니다.
        ```python
        # 변경 전: speaker_url = "http://[Ubuntu_IP_Address]:8000/notify"
        speaker_url = "[http://192.168.1.101:8000/notify](http://192.168.1.101:8000/notify)" # 예시
        ```
    * **`human_detect_buzzer.py` (라즈베리파이에서 실행될 파일):**
        `send_to_speaker` 함수 내의 `speaker_url` 변수를 Ubuntu PC의 실제 IP 주소로 변경합니다.
        ```python
        # 변경 전: speaker_url = "http://[Ubuntu_IP_Address]:8000/notify"
        speaker_url = "[http://192.168.1.101:8000/notify](http://192.168.1.101:8000/notify)" # 예시
        ```

### 💻 스크립트 실행 순서

모든 스크립트는 해당 장치의 프로젝트 디렉토리(`/path/to/your/project_ai/`)로 이동한 후 실행해야 합니다.

1.  **Ubuntu PC에서 `local_sever.py` 실행:**
    가장 먼저 실행하여 Ollama LLM 서버 및 TTS 음성 피드백 서버를 활성화합니다.
    ```bash
    cd /path/to/your/project_ai/
    python3 local_sever.py
    ```
    * _`ollama run gemma:2b` 명령어를 통해 `Ollama` 서버가 미리 백그라운드에서 실행 중이어야 합니다._

2.  **라즈베리파이에서 `human_detect_buzzer.py` 실행:**
    사람 감지 및 부저/음성 알림 기능을 담당합니다.
    ```bash
    cd /path/to/your/project_ai/
    sudo python3 human_detect_buzzer.py
    ```
    * `sudo` 권한이 필요합니다 (카메라 및 GPIO 접근).

3.  **라즈베리파이에서 `main_runner.py` 실행:**
    `gpio_server.py`와 `gesture_debounce_success.py`를 함께 실행합니다. `gpio_server.py`는 하드웨어 제어, `gesture_debounce_success.py`는 손 제스처 인식을 담당합니다.
    ```bash
    cd /path/to/your/project_ai/
    sudo python3 main_runner.py
    ```
    * 이 명령을 실행하면 `gpio_server.py`와 `gesture_debounce_success.py`가 자동으로 백그라운드에서 실행됩니다.
    * `sudo` 권한이 필요합니다 (GPIO 및 카메라 접근).
    * `Ctrl+C`를 누르면 `main_runner.py`가 실행 중인 모든 서브스크립트를 안전하게 종료합니다.

---
