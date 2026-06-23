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

- **점수 분석용 세부 지표** (`pipeline/scorer.py`)
  - `similarity_score` — penalty 적용 전 base_score를 진단용 필드로 details에 노출
  - `mfcc_score_used` — 실제 base_score 계산에 사용된 mfcc_score 값 기록
  - `penalty_breakdown` — 항목별 감점 값을 dict로 묶어 일괄 확인 가능하게 추가
  - 채점 결과(`final_score`)에는 영향 없음

- **C0 제외 MFCC 비교 지표** (`pipeline/scorer.py`)
  - `effective_mfcc_score` — `mfcc_score * 0.4 + mfcc_no_c0_score * 0.6` 혼합 점수를 details에 기록
  - `effective_mfcc_strategy` — 계산 방식 식별자(`mfcc_40_no_c0_60` 또는 `mfcc_only`) 기록
  - mfcc_no_c0_score가 없으면 mfcc_score 그대로 사용하고 strategy를 `mfcc_only`로 표시
  - 채점 결과(`final_score`)에는 영향 없음 — 향후 Stage 3에서 실제 가중치에 반영 예정

- **MFCC C0 제외 점수 diagnostic** (`pipeline/scorer.py`, `scripts/evaluate_labeled_recordings.py`)
  - `z_score_distance_score_without_c0()` 함수 추가
  - 각 scoring 함수(`score_vowel`, `score_consonant`, `score_liquid` 등)에서 `mfcc_no_c0_score`, `mfcc_c0_score_gap`을 `details`에 함께 기록
  - C0는 전체 spectral energy 성격이 강해 pitch·개인 음색 차이에 민감하므로, 이를 제외했을 때 점수 차이를 데이터로 확인하기 위한 실험용
  - `final_score` 계산에는 기존 `mfcc_score`를 그대로 사용 (채점 결과에 영향 없음)
  - `evaluate_labeled_recordings.py` `DETAIL_FIELDS`에 `mfcc_no_c0_score`, `mfcc_c0_score_gap` 추가

---

<!-- 새 항목은 위에 추가 -->
