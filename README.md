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

발음 점수는 사용자 음성에서 추출한 acoustic feature를 원어민 reference vector와 비교해 계산합니다.
단, 모든 음소를 같은 기준으로 평가하지 않고, **음소별로 실제 발음 차이를 가장 잘 드러내는 feature에 더 높은 가중치**를 둡니다.

### 음소 유형별 가중치

| 유형                    | MFCC | Duration | Centroid | ZCR | RMS | 기준을 이렇게 둔 이유                                                   |
| --------------------- | ---: | -------: | -------: | --: | --: | -------------------------------------------------------------- |
| 모음                    |  70% |      15% |      10% |   — |  5% | 모음은 입 모양과 혀 위치에 따른 **음색 차이**가 가장 중요하므로 MFCC를 중심으로 평가합니다.       |
| 지속시간 모음 `/i/`, `/iː/` |  50% |      35% |      10% |   — |  5% | 짧은 모음과 긴 모음의 구분에는 **길이 차이**가 중요하므로 Duration 비중을 높였습니다.         |
| 자음                    |  55% |        — |      10% | 35% |   — | 마찰음과 무성 자음은 **마찰·고주파 특성**이 중요하므로 ZCR과 Centroid를 함께 반영합니다.      |
| 유음 `/r/`, `/l/`       |  75% |      15% |      10% |   — |   — | `/r/`, `/l/`은 마찰음이 아니기 때문에 ZCR보다 **조음 위치와 음색 변화**를 더 중요하게 봅니다. |

> MFCC는 발음의 전체적인 음색과 조음 차이를 나타내는 핵심 feature입니다.
> Duration, Centroid, ZCR, RMS는 음소별 특성을 보완하기 위한 보조 feature로 사용합니다.

---

### 패널티 항목

기본 점수는 원어민 reference와의 유사도를 기준으로 계산하고, 특정 한국어식 오류 패턴이 감지될 때만 추가 패널티를 적용합니다.

| 항목               | 기준                                                     | 최대 감점 | 적용 이유                                                              |
| ---------------- | ------------------------------------------------------ | ----: | ------------------------------------------------------------------ |
| 한국어식 패턴 패널티      | EN reference와 KO reference의 MFCC 상대 거리 비교              |  -15점 | 사용자의 발음이 원어민 기준보다 한국어식 오류 reference에 가까운 경우를 보정하기 위한 패널티입니다.       |
| `/f/` 마찰음 오독 패널티 | Onset ZCR 비율, Onset MFCC 거리, RMS Crest Factor 교차 검증    |   -6점 | `/f/`는 단어 초반의 마찰음이 핵심이므로, 전체 평균이 아니라 onset 구간의 마찰 특성을 별도로 확인합니다.   |
| 유음 음향 패널티        | `/r/`은 mel F3 계열 비율, `/l/`은 MFCC 전이 거리·C0 delta 기반     | -9.5점 | `/r/`, `/l/`은 한국어식 `ㄹ`로 뭉개지기 쉬워, 단어 전체 점수와 별도로 유음 전이 특성을 추가 확인합니다. |
| `/ə/` 과강세 패널티    | Onset RMS가 reference onset RMS 및 전체 RMS 대비 과도하게 강한지 확인 |  -12점 | schwa `/ə/`는 약하게 지나가야 하므로, 첫 음절이 한국어식으로 또렷하고 강하게 발음되는 경우를 감점합니다.   |

[자세한 채점공식](Pronunciation_Scoring_Design.md)

---

### 입력 검증 Gate

발음 점수를 계산하기 전에 먼저 입력 음성이 평가 가능한 상태인지 확인합니다.

* 녹음 품질이 낮거나 음성이 너무 짧은 경우에는 pronunciation score를 계산하지 않습니다.
* faster-whisper 기반 target word validation을 통해 목표 단어와 다른 발화로 판단되면 word mismatch로 처리합니다.
* 정상 입력으로 판단된 경우에만 원어민 reference와 비교해 발음 점수를 계산합니다.

이를 통해 **발음 점수**, **단어 일치 여부**, **녹음 품질 문제**를 분리하여 결과를 해석할 수 있도록 설계했습니다.


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

---

## 향후 개선 방향

현재 MVP는 원어민 reference와 사용자 발음을 비교해 발음 점수를 계산합니다.  
다만 음소마다 중요한 발음 단서가 다르기 때문에, 이후 버전에서는 음소별로 다른 분석 기준을 추가할 예정입니다.

| 음소 | 현재 평가 기준 | 고도화 방향 |
|---|---|---|
| `/i/` | MFCC와 duration 중심 | 짧게 끝나는 모음인지 확인하기 위해 `vowel sustain` 구간을 분석합니다. |
| `/iː/` | MFCC와 duration 중심 | 단순 길이뿐 아니라 모음 에너지가 안정적으로 오래 유지되는지 확인합니다. |
| `/r/` | MFCC, duration, centroid + 유음 acoustic penalty | `road/load`, `rice/lice` 같은 최소쌍 데이터를 늘려 `/r/` 특유의 음색과 전이 패턴을 더 안정적으로 검증합니다. |
| `/l/` | MFCC, duration, centroid + 유음 acoustic penalty | 모음으로 넘어가는 전이 구간을 더 세밀하게 분석해 한국어식 `ㄹ` 발음과 구분합니다. |
| `/f/` | 자음 feature + onset 마찰음 보조 패널티 | 단어 초반 마찰음이 충분히 살아 있는지 onset ZCR, MFCC, RMS crest factor로 계속 검증합니다. |
| `/v/` | 자음 feature + 한국어식 reference penalty | `/f/`와 달리 성대 울림이 있는 유성 마찰음이므로, 마찰음과 voicing을 함께 보는 전용 feature를 추가할 예정입니다. |
| `/θ/` | 자음 feature + 한국어식 reference penalty | 혀끝 마찰음이 `/s/`, `/t/`처럼 바뀌는 패턴을 더 잘 구분하도록 onset 마찰 특성을 추가 검토합니다. |
| `/æ/` | 모음 feature + 한국어식 reference penalty | 입을 충분히 벌린 밝은 모음인지 확인하기 위해 MFCC와 centroid 기반 분리를 더 검증합니다. |
| `/ə/` | 모음 feature + 과강세 패널티 | 첫 음절을 너무 강하게 발음하는지 onset RMS 기준을 개선하고, 약화 모음 특성을 더 안정적으로 판단합니다. |
| `/oʊ/` | 모음 feature 중심 | 단일 모음처럼 들리는지, 뒤로 미끄러지는 diphthong 특성이 유지되는지 추가 분석합니다. |

위 개선 방향은 단순히 feature를 많이 추가하는 것이 아니라, 각 음소에서 실제로 발음 차이를 만드는 단서를 분리하기 위한 작업입니다.  
특히 `/i/`와 `/iː/`는 단어 전체 길이보다 모음이 유지되는 구간을, `/r/`와 `/l/`은 최소쌍 데이터를 통한 전이 패턴을, `/v/`는 마찰음과 성대 울림을 함께 보는 방향으로 고도화할 예정입니다.