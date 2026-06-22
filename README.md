# PronounceAI

영어를 배우는 한국인을 위한 **발음 채점 웹 앱**입니다.
마이크로 단어를 녹음하면 MFCC·ZCR·Duration·RMS 등 음향 특징을 원어민 레퍼런스와 비교해 0–100점으로 채점하고, 한국어식 발음 패턴을 감지해 구체적인 피드백을 제공합니다.

---

## 파이프라인

![PronounceAI 발음 채점 파이프라인]
<p align="center">
<img width="1672" height="941" alt="Image" src="https://github.com/user-attachments/assets/03113d51-af21-4064-9d4a-c7197154c8d2" />
</p>

| 단계 | 설명 |
|------|------|
| **브라우저 녹음** | 사용자가 타겟 단어를 마이크로 녹음 |
| **오디오 업로드** | `POST /api/pronunciation/analyze` (FastAPI) |
| **오디오 전처리** | librosa load → VAD 묵음 제거 |
| **특징 추출** | MFCC(13차), ZCR, Duration, RMS, Spectral Centroid, Onset 특징 |
| **녹음 품질 검사** | Whisper STT 단어 일치 확인 + 음량·노이즈·길이 필터 |
| **발음 채점** | 음소 유형별(모음/자음/유음) 가중치 스코어링 |
| **패턴 패널티** | 한국어식 reference 비교 / 유음(/r/, /l/) 음향 분석 / /ə/ 과강세 탐지 |
| **결과 반환** | 점수·피드백·세부 지표 JSON |
| **UI 표시** | 레이더 차트·히스토리·지표별 점수 |

---

## 지원 음소

| 음소 | 예시 단어 | 특이 처리 |
|------|-----------|-----------|
| `/θ/` | think, thick, thin | 한국어식 ㄷ/ㅅ 혼동 감지 |
| `/f/` | fan, fix, fox | Onset ZCR·MFCC·Crest Factor 3중 교차 검증 |
| `/v/` | van, vine, void | - |
| `/i/` | ship, bit, sit | 지속 시간 가중치 강화 (짧고 가볍게) |
| `/iː/` | sheep, beat, seat | 모음 길이 비교 |
| `/æ/` | bad, cat, man | - |
| `/ə/` | about, again, away | Onset RMS 과강세 패널티 |
| `/oʊ/` | go, note, home | - |
| `/r/` | rice, road, rock | F3 극저값 핀셋 패널티, 액체음 Onset 분석 |
| `/l/` | lice, load, lock | Borderline 액체음 음향 패널티 |

---

## 채점 공식

### 음소 유형별 가중치

| 유형 | MFCC | Duration | Centroid | ZCR | RMS |
|------|------|----------|----------|-----|-----|
| 모음 | 70 % | 15 % | 10 % | — | 5 % |
| 지속시간 모음 (`/i/`) | 50 % | 35 % | 10 % | — | 5 % |
| 자음 | 55 % | — | 10 % | 35 % | — |
| 유음 (`/r/`, `/l/`) | 75 % | 15 % | 10 % | — | — |

### 패널티 항목

- **한국어식 패턴 패널티**: EN reference vs KO reference MFCC 상대 거리, 최대 −15점
- **/f/ 마찰음 오독**: Onset ZCR 비율·Onset MFCC 거리·RMS Crest Factor 교차 검증, 최대 −6점
- **유음 음향 패널티**: /r//l/ MFCC 전이 거리 기반, 최대 −10점
- **/ə/ 과강세**: Onset RMS가 전체 RMS 대비 과도하게 강할 때 감점, 최대 −12점

---

## 프로젝트 구조

```
pronunciation_design/
├── main.py                  # FastAPI 앱 진입점
├── pipeline/
│   ├── audio.py             # 오디오 로드 & 묵음 제거
│   ├── features.py          # 특징 추출 (MFCC, ZCR, RMS, ...)
│   ├── liquid_features.py   # /r/, /l/ 전용 음향 특징
│   ├── scorer.py            # 채점 & 패널티 로직
│   ├── quality.py           # 녹음 품질 검사 (Whisper STT)
│   ├── reference.py         # 레퍼런스 벡터 로드
│   ├── compare.py           # EN vs KO 비교 분석
│   ├── db.py                # 녹음 결과 저장
│   └── word_targets.py      # 단어 타겟 정의
├── webapp/
│   ├── routes/
│   │   ├── pages.py         # HTML 페이지 라우터
│   │   └── pronunciation_api.py  # REST API 엔드포인트
│   ├── services/
│   │   └── pronunciation_facade.py
│   └── schemas/
├── data/
│   ├── words.txt            # 단어·음소·한국어 발음 목록
│   ├── reference_vectors.json    # 원어민 영어 레퍼런스 벡터
│   ├── ko_reference_vectors.json # 한국어식 레퍼런스 벡터
│   ├── reference_en/        # 원어민 영어 음성 샘플
│   └── reference_ko/        # 한국어식 음성 샘플
├── scripts/
│   ├── build_reference.py   # 레퍼런스 벡터 빌드
│   ├── build_gtts.py        # TTS 샘플 생성
│   └── evaluate_labeled_recordings.py  # 라벨링 결과 평가
├── templates/               # Jinja2 HTML 템플릿
├── static/                  # CSS, JS
└── pipeline.png             # 파이프라인 다이어그램
```

---

## 실행 방법

### 요구사항

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

### 설치 & 실행

```bash
# 의존성 설치
uv sync

# 서버 실행
uv run python main.py
```

브라우저에서 `http://localhost:8000` 접속.

### 환경변수

`.env` 파일에 아래 값을 설정합니다.

```
ELEVENLABS_API_KEY=your_key_here
```

---

## 레퍼런스 벡터 빌드

```bash
# TTS(gTTS)로 영어 샘플 생성
uv run python scripts/build_gtts.py

# 레퍼런스 벡터 계산
uv run python scripts/build_reference.py
```

---

## 기술 스택

| 영역 | 라이브러리 |
|------|-----------|
| 웹 프레임워크 | FastAPI, Uvicorn, Jinja2 |
| 오디오 분석 | librosa, soundfile, pydub |
| STT (품질 검사) | faster-whisper |
| ML | numpy, scikit-learn |
| 시각화 | Plotly.js (프론트), Plotly (백엔드) |
| TTS (레퍼런스 생성) | gTTS, ElevenLabs |
