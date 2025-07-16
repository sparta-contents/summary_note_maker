# 📝 SRT 요약 노트 생성기

Google Drive에 있는 `.srt` 자막 파일을 읽어 Gemini AI로 내용을 분석하고, 구조화된 JSON 형식의 요약 노트를 생성하는 Streamlit 웹 애플리케이션입니다.

## ✨ 주요 기능

-   **Google Drive 연동**: Google Drive 폴더 URL을 입력하여 폴더 내의 모든 `.srt` 파일을 자동으로 탐색합니다.
-   **AI 기반 자동 요약**: Google Gemini AI를 사용하여 각 자막 파일의 내용을 주제별, 시간대별로 정리된 요약 노트를 생성합니다.
-   **JSON 형식 출력**: 생성된 요약 노트는 계층적 구조를 가진 JSON 형식으로 제공됩니다.
-   **개별 및 일괄 처리**: 단일 파일을 선택하여 처리하거나, 폴더 내의 모든 SRT 파일을 한 번에 처리하여 각각의 JSON 결과물을 생성할 수 있습니다.
-   **결과물 저장**: 생성된 JSON 파일은 로컬에 다운로드하거나, 원본 파일이 있던 Google Drive 폴더 내의 '요약노트'라는 하위 폴더에 자동으로 저장할 수 있습니다.

---

## 🚀 실행 방법

### 1. 사전 준비

#### 가. Google Cloud 설정

1.  **Google Cloud Platform (GCP) 프로젝트**를 생성하거나 기존 프로젝트를 선택합니다.
2.  **Google Drive API**와 **Vertex AI API** (또는 Generative Language API)를 활성화합니다.
3.  **서비스 계정(Service Account)**을 생성합니다.
    -   생성된 서비스 계정에 `Google Drive API`에 접근할 수 있는 역할(예: `편집자`)을 부여합니다.
    -   서비스 계정 키(JSON 형식)를 생성하고 다운로드합니다.
4.  다운로드한 서비스 계정 키 파일의 이름을 `credentials.json`으로 변경하고, 이 프로젝트의 루트 디렉터리에 위치시킵니다.

#### 나. Gemini API 키 발급

1.  [Google AI Studio](https://aistudio.google.com/app/apikey)에 방문하여 Gemini API 키를 발급받습니다.

### 2. 프로젝트 설정

#### 가. Streamlit 인증 정보 설정

1.  프로젝트 루트 디렉터리에 `.streamlit`이라는 이름의 폴더를 생성합니다.
2.  `.streamlit` 폴더 내에 `secrets.toml` 파일을 생성합니다.
3.  아래 내용을 `secrets.toml` 파일에 복사한 뒤, `YOUR_GEMINI_API_KEY` 부분을 위에서 발급받은 자신의 Gemini API 키로 교체합니다.

    **`.streamlit/secrets.toml`**:
    ```toml
    [api_keys]
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
    ```

#### 나. 가상 환경 및 라이브러리 설치

터미널을 열고 프로젝트 폴더로 이동한 뒤, 아래 명령어를 순서대로 실행합니다.

```bash
# 1. 가상 환경 생성 (최초 1회)
python -m venv venv

# 2. 가상 환경 활성화 (Windows 기준)
.\venv\Scripts\activate

# 3. 필요한 라이브러리 설치
pip install -r requirements.txt
```

### 3. 애플리케이션 실행

가상 환경이 활성화된 상태에서 아래 명령어를 실행합니다.

```bash
streamlit run app.py
```

브라우저에서 앱이 열리면, Google Drive 폴더 URL을 입력하고 "폴더의 SRT 파일 목록 가져오기" 버튼을 클릭하여 시작할 수 있습니다.