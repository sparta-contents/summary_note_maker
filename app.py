import streamlit as st
import pandas as pd
import re
import os
import json
import io
import uuid
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import google.generativeai as genai

# --- API 및 서비스 계정 설정 ---

# Streamlit 배포 환경에서는 secrets에서 자격 증명 로드
SERVICE_ACCOUNT_FILE = "credentials.json"
if "GOOGLE_CREDENTIALS_JSON" in st.secrets:
    # Streamlit Cloud의 secrets에서 JSON 문자열을 가져와 파일로 씀
    creds_json_str = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    with open(SERVICE_ACCOUNT_FILE, "w") as f:
        f.write(creds_json_str)
else:
    st.error("오류: Streamlit secrets에 GOOGLE_CREDENTIALS_JSON이 없습니다. .streamlit/secrets.toml 파일을 확인하세요.")
    st.stop()

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

# Streamlit Secrets에서 Gemini API 키 가져오기
try:
    GEMINI_API_KEY = st.secrets["api_keys"]["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, AttributeError):
    st.error("오류: .streamlit/secrets.toml 파일에 Gemini API 키가 설정되지 않았습니다.")
    st.info("`[api_keys]` 섹션 아래에 `GEMINI_API_KEY='...'` 형식으로 키를 추가해주세요.")
    st.stop()

# --- Google Drive 관련 함수 ---

@st.cache_resource
def get_drive_service():
    """서비스 계정으로 Google Drive API 서비스 객체를 생성합니다."""
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=DRIVE_SCOPES
        )
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        st.error(f"Drive 서비스 생성 중 오류: {e}")
        return None

def extract_folder_id_from_url(url: str):
    """Google Drive URL에서 폴더 ID를 추출합니다."""
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if match: return match.group(1)
    match = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if match: return match.group(1)
    return None

def list_files_in_folder(service, folder_id: str):
    """폴더 안의 파일 목록을 가져옵니다."""
    try:
        query = f"'{folder_id}' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'"
        results = service.files().list(
            q=query, corpora='allDrives', includeItemsFromAllDrives=True,
            supportsAllDrives=True, pageSize=500,
            fields="files(id, name, mimeType)"
        ).execute()
        return results.get("files", []), None
    except Exception as e:
        return None, str(e)

def download_srt_file(service, file_id: str) -> str:
    """SRT 파일 내용을 문자열로 다운로드합니다."""
    try:
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return file_io.getvalue().decode('utf-8', errors='ignore')
    except Exception:
        return None

def get_or_create_folder(service, parent_folder_id: str, new_folder_name: str):
    """지정된 부모 폴더 안에 폴더가 없으면 생성하고 ID를 반환합니다."""
    query = (f"name='{new_folder_name}' and mimeType='application/vnd.google-apps.folder' "
             f"and '{parent_folder_id}' in parents and trashed=false")
    response = service.files().list(q=query, corpora='allDrives', includeItemsFromAllDrives=True,
                                    supportsAllDrives=True, fields='files(id)').execute()
    if response.get('files'):
        return response['files'][0]['id']
    else:
        file_metadata = {'name': new_folder_name, 'mimeType': 'application/vnd.google-apps.folder',
                         'parents': [parent_folder_id]}
        folder = service.files().create(body=file_metadata, supportsAllDrives=True, fields='id').execute()
        return folder.get('id')

def upload_json_to_drive(service, folder_id: str, file_name: str, json_data: str):
    """JSON 데이터를 Google Drive의 지정된 폴더에 업로드합니다."""
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    media = MediaIoBaseUpload(io.BytesIO(json_data.encode('utf-8')), mimetype='application/json')
    file = service.files().create(body=file_metadata, media_body=media,
                                  supportsAllDrives=True, fields='id, webViewLink').execute()
    return file.get('webViewLink')

def parse_srt_to_transcript(srt_content: str) -> str:
    """SRT 파일 내용을 '시간: 대사' 형식의 문자열로 변환합니다."""
    if not srt_content: return ""
    
    def time_to_seconds(t):
        h, m, s_ms = t.split(':')
        s, ms = s_ms.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    lines = srt_content.strip().split('\n\n')
    transcript_lines = []
    for line in lines:
        parts = line.split('\n')
        if len(parts) >= 3:
            try:
                time_line = parts[1]
                start_time_str = time_line.split(' --> ')[0]
                start_time_sec = time_to_seconds(start_time_str)
                text = " ".join(parts[2:]).strip()
                transcript_lines.append(f"[{start_time_sec:.2f}s] {text}")
            except (ValueError, IndexError):
                continue # 잘못된 형식의 라인 건너뛰기
    return "\n".join(transcript_lines)


# --- Gemini AI 관련 함수 ---
@st.cache_data(show_spinner=False)
def generate_structured_notes_with_gemini(_srt_content: str, file_name: str):
    """Gemini AI를 사용하여 SRT 자막으로 구조화된 JSON 요약 노트를 생성합니다. (후처리 전)"""
    transcript = parse_srt_to_transcript(_srt_content)
    if not transcript:
        return [{"error": "SRT 내용이 비어있거나 파싱할 수 없습니다."}]

    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    당신은 전문 강의 요약 노트 작성자입니다. 제공되는 SRT 자막 파일 내용을 분석하여, 주제별로 내용을 정리하고 계층적 구조를 가진 JSON 형식의 요약 노트를 생성해야 합니다.

    다음은 '{file_name}' 파일의 타임스탬프가 포함된 자막 내용입니다.
    ---
    {transcript}
    ---

    **요구사항:**
    1.  **주제별 그룹화:** 전체 자막 내용을 의미적으로 연결되는 여러 'section'으로 나눕니다.
    2.  **계층 구조:** 각 섹션의 주제에 따라 `level`을 1 또는 2로 설정하여 목차 구조를 만듭니다. (주로 level 1 사용)
    3.  **제목 (title):** 각 섹션의 핵심 주제를 나타내는 간결하고 명확한 제목을 작성합니다. (예: "🤖 AI 로봇의 활용 예시와 가능성") 적절하게 이모지를 사용해주세요.
    4.  **내용 (content):** 각 섹션의 핵심 내용을 설명체 문장으로 요약하여 배열에 담습니다. 원문 내용을 바탕으로 자연스럽게 문장을 다듬어주세요. 내용은 여러개의 불릿으로 정리해주세요.
    5.  **시작 시간 (startTime):** 각 섹션을 구성하는 자막 중, 가장 먼저 시작하는 자막의 시간(초)을 `startTime` 값으로 입력합니다.
    6.  **레이아웃 (layout):** 개념 설명이나 도구 소개 등 정보 나열은 'bulletList'로, 흐름이 있는 서사는 'paragraph'로 지정합니다.
    7.  **고유 ID (attrs.id):** 각 섹션마다 "GENERATE_UUID"라는 플레이스홀더를 `attrs.id`에 할당합니다. 실제 값은 나중에 채워집니다.
    8.  **기타 속성:** `type`은 "section", `trigger`는 "timeline"으로 고정하고, `1-2.json` 예시 파일의 `attrs` 구조를 최대한 따릅니다. `chunkindex`는 생성하지 마세요.
    9. \ 는 문장에 포함되지 않도록 해주세요.
    **출력 JSON 형식 예시:**
    ```json
    [
        {{
            "type": "section",
            "content": [
                ""AI, 나도 할 수 있다"라는 주제로 강의가 시작되며, AI의 개념을 보다 쉽게 이해할 수 있도록 다양한 예시를 함께 살펴봅니다. ",
                "AI 로봇은 **물건 정리와 분류**를 스스로 수행하며, 직관적으로 사용하실 수 있는 기능을 갖추고 있습니다."
            ],
            "title": "1. 🤖 AI 로봇의 활용 예시와 가능성",
            "level": 1,
            "startTime": 0.0,
            "attrs": {{
                "id": "GENERATE_UUID",
                "display": "none",
                "color": "",
                "loading": false,
                "layout": "bulletList",
                "trigger": "timeline",
                "sentenceIndices": "",
                "data-value": ""
            }}
        }}
    ]
    ```

    이제 위 요구사항과 예시 형식에 맞춰 JSON을 생성해주세요. JSON 코드 블록만 출력하고 다른 설명은 생략해주세요.
    """
    try:
        response = model.generate_content(prompt)
        cleaned_response = re.sub(r'```json\s*|\s*```', '', response.text, flags=re.DOTALL).strip()
        
        try:
            data = json.loads(cleaned_response)
            # UUID 할당 및 백슬래시 제거
            for item in data:
                if isinstance(item, dict):
                    if item.get('type') == 'section':
                        if 'attrs' not in item:
                            item['attrs'] = {}
                        item['attrs']['id'] = str(uuid.uuid4())
                    
                    # title 값에 포함된 불필요한 백슬래시 제거
                    if 'title' in item and isinstance(item.get('title'), str):
                        item['title'] = item['title'].replace('', '')
                    
                    # content 리스트의 각 문자열에서 불필요한 백슬래시 제거
                    if 'content' in item and isinstance(item.get('content'), list):
                        item['content'] = [
                            line.replace('', '') if isinstance(line, str) else line
                            for line in item['content']
                        ]
            return data # 파이썬 객체 반환
        except json.JSONDecodeError:
            return [{"error": "AI가 유효하지 않은 JSON을 생성했습니다.", "raw_response": cleaned_response}]

    except Exception as e:
        return [{"error": f"AI 요약 생성 중 오류 발생: {e}"}]


# --- Streamlit UI ---
st.set_page_config(page_title="SRT 요약 노트 생성기", page_icon="📝")
st.title("📝 SRT 파일 상세 요약 노트 생성기")

if 'current_folder_id' not in st.session_state:
    st.session_state.current_folder_id = None
if 'drive_files' not in st.session_state:
    st.session_state.drive_files = []
if 'generated_json' not in st.session_state:
    st.session_state.generated_json = None
if 'selected_file_name' not in st.session_state:
    st.session_state.selected_file_name = None


drive_service = get_drive_service()

if drive_service:
    drive_url = st.text_input("파일 목록을 조회할 Google Drive 폴더 URL을 입력하세요",
                            placeholder="https://drive.google.com/drive/folders/...")

    if st.button("폴더의 SRT 파일 목록 가져오기", use_container_width=True):
        st.session_state.generated_json = None # 목록 새로고침 시 이전 결과 초기화
        folder_id = extract_folder_id_from_url(drive_url) if drive_url else None
        if folder_id:
            st.session_state.current_folder_id = folder_id
            with st.spinner("파일 목록을 가져오는 중..."):
                files, error = list_files_in_folder(drive_service, folder_id)
                if error:
                    st.error(f"파일 목록 조회 실패: {error}")
                    st.session_state.drive_files = []
                else:
                    srt_files = [f for f in files if f['name'].lower().endswith('.srt')]
                    st.session_state.drive_files = srt_files
                    if not srt_files:
                        st.warning("폴더에 .srt 파일이 없습니다.")
                    else:
                        st.success(f"총 {len(srt_files)}개의 SRT 파일을 찾았습니다.")
        else:
            st.warning("유효한 Google Drive 폴더 URL을 입력해주세요.")
            st.session_state.drive_files = []

    if st.session_state.drive_files:
        srt_files = st.session_state.drive_files

        file_options = {f['name']: f['id'] for f in srt_files}
        
        selected_file_name = st.radio(
            "요약 노트를 생성할 파일을 선택하세요.",
            options=list(file_options.keys()),
            index=None,
            key="file_selector"
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("선택한 파일로 요약 노트 생성", use_container_width=True, disabled=not selected_file_name):
                st.session_state.selected_file_name = selected_file_name
                file_id = file_options[selected_file_name]
                with st.spinner(f"'{selected_file_name}' 파일 처리 중..."):
                    content = download_srt_file(drive_service, file_id)
                    if content:
                        # 1. AI로 노트 생성 (후처리 전)
                        notes_data = generate_structured_notes_with_gemini(content, selected_file_name)
                        
                        # 2. 후처리: chunkindex 할당
                        for i, item in enumerate(notes_data):
                            if isinstance(item, dict) and item.get('type') == 'section':
                                item['chunkindex'] = [i]
                                if 'attrs' in item:
                                    item['attrs']['chunkindex'] = i

                        st.session_state.generated_json = json.dumps(notes_data, indent=2, ensure_ascii=False)
                    else:
                        st.error("파일 내용을 가져오는 데 실패했습니다.")
                        st.session_state.generated_json = None

        with col2:
            if st.button("모든 SRT 파일 한 번에 처리", type="primary", use_container_width=True):
                # 이전 작업 결과 초기화
                st.session_state.generated_json = None
                st.session_state.selected_file_name = None

                parent_folder_id = st.session_state.get('current_folder_id')
                if not parent_folder_id:
                    st.error("현재 폴더 ID를 찾을 수 없어 작업을 시작할 수 없습니다.")
                else:
                    with st.spinner("'요약노트' 폴더 확인 및 생성 중..."):
                        target_folder_id = get_or_create_folder(drive_service, parent_folder_id, "요약노트")

                    results_container = st.container()
                    progress_bar = st.progress(0, text="전체 요약 작업 준비 중...")
                    total_files = len(file_options)

                    for i, (file_name, file_id) in enumerate(file_options.items()):
                        progress_text = f"({i+1}/{total_files}) '{file_name}' 처리 중..."
                        progress_bar.progress((i + 1) / total_files, text=progress_text)

                        content = download_srt_file(drive_service, file_id)
                        if not content:
                            results_container.warning(f"⚠️ '{file_name}' 파일 내용을 가져오지 못해 건너뜁니다.")
                            continue

                        # 1. AI로 노트 생성
                        notes_data = generate_structured_notes_with_gemini(content, file_name)
                        if isinstance(notes_data, list) and notes_data and 'error' in notes_data[0]:
                            results_container.error(f"❌ '{file_name}' 요약 생성 실패: {notes_data[0].get('raw_response', notes_data[0]['error'])}")
                            continue
                        
                        # 2. 후처리: chunkindex 할당
                        for chunk_idx, item in enumerate(notes_data):
                            if isinstance(item, dict) and item.get('type') == 'section':
                                item['chunkindex'] = [chunk_idx]
                                if 'attrs' in item:
                                    item['attrs']['chunkindex'] = chunk_idx
                        
                        # 3. JSON 데이터 준비 및 업로드
                        json_data = json.dumps(notes_data, indent=2, ensure_ascii=False)
                        output_file_name = os.path.splitext(file_name)[0] + ".json"

                        try:
                            file_link = upload_json_to_drive(drive_service, target_folder_id, output_file_name, json_data)
                            results_container.success(f"✅ '{output_file_name}' 생성 완료! [Google Drive에서 보기]({file_link})")
                        except Exception as e:
                            results_container.error(f"❌ '{output_file_name}' 업로드 실패: {e}")

                    progress_bar.empty()
                    st.success("모든 파일 처리가 완료되었습니다.")
    
    if st.session_state.get('generated_json'):
        st.success(f"요약 노트 생성이 완료되었습니다!")
        
        json_data = st.session_state.generated_json
        
        # 파일 이름 결정
        if st.session_state.selected_file_name == "all_files_summary":
            output_file_name = "summary_note_all.json"
        else:
            output_file_name = os.path.splitext(st.session_state.selected_file_name)[0] + ".json"

        st.code(json_data, language='json')

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(label="💾 로컬에 JSON으로 다운로드", data=json_data,
                               file_name=output_file_name, mime="application/json",
                               use_container_width=True)
        with dl_col2:
            if st.button("☁️ Google Drive에 저장", use_container_width=True):
                parent_folder_id = st.session_state.get('current_folder_id')
                if parent_folder_id:
                    with st.spinner("'요약노트' 폴더 확인 및 생성 중..."):
                        target_folder_id = get_or_create_folder(drive_service, parent_folder_id, "요약노트")
                    
                    with st.spinner(f"`{output_file_name}` 파일 업로드 중..."):
                        file_link = upload_json_to_drive(drive_service, target_folder_id, output_file_name, json_data)
                    
                    st.success("Google Drive에 성공적으로 저장되었습니다!")
                    st.link_button("저장된 파일 보기", url=file_link)
                else:
                    st.error("현재 폴더 ID를 찾을 수 없어 저장할 수 없습니다.")
else:
    st.warning("Google Drive 서비스에 연결할 수 없습니다. `credentials.json` 파일이 올바른지 확인하세요.")
