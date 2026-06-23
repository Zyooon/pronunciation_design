# Changelog

변경사항은 최신순으로 기록한다.

---

## [v2] - 2026-06-23

### 삭제

- **미사용 함수 제거** (`pipeline/audio.py`, `pipeline/quality.py`, `webapp/services/pronunciation_facade.py`)
  - `get_duration_ms` — `pipeline/features.py`의 `extract_duration_ms`와 역할이 중복되어 미사용 상태
  - `check_word_match` — `check_word_match_detail`의 호환 래퍼였으나 실제 호출부 없음
  - `analyze_audio` — `analyze_audio_with_features`의 래퍼였으나 import하는 곳 없음

### 추가

- **MFCC C0 제외 점수 diagnostic** (`pipeline/scorer.py`, `scripts/evaluate_labeled_recordings.py`)
  - `z_score_distance_score_without_c0()` 함수 추가
  - 각 scoring 함수(`score_vowel`, `score_consonant`, `score_liquid` 등)에서 `mfcc_no_c0_score`, `mfcc_c0_score_gap`을 `details`에 함께 기록
  - C0는 전체 spectral energy 성격이 강해 pitch·개인 음색 차이에 민감하므로, 이를 제외했을 때 점수 차이를 데이터로 확인하기 위한 실험용
  - `final_score` 계산에는 기존 `mfcc_score`를 그대로 사용 (채점 결과에 영향 없음)
  - `evaluate_labeled_recordings.py` `DETAIL_FIELDS`에 `mfcc_no_c0_score`, `mfcc_c0_score_gap` 추가

---

<!-- 새 항목은 위에 추가 -->
