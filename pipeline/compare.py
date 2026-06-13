import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.db import DEFAULT_DB_PATH


ReferenceVector = dict[str, float | list[float] | str]
FeatureDict = dict[str, Any]

_COMPARISON_METRICS = (
    ("duration_ms", "Duration", "ms", "duration_ms"),
    ("zcr_mean", "ZCR", "", "zcr_mean"),
    ("rms_mean", "RMS", "", "rms_mean"),
    ("spectral_centroid_mean", "Spectral centroid", "Hz", "spectral_centroid_mean"),
)


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "-"
    if unit == "ms":
        return f"{value:.0f}ms"
    if unit == "Hz":
        return f"{value:.0f}Hz"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"


def _percent_width(value: float | None, ref_value: float | None) -> float:
    if value is None or ref_value is None:
        return 0.0
    scale = max(abs(value), abs(ref_value), 1e-8)
    return max(0.0, min(100.0, abs(value) / scale * 100))


def compute_mfcc_distance(user_mfcc: Any, reference_mfcc: Any) -> float | None:
    user_values = _safe_list(user_mfcc)
    ref_values = _safe_list(reference_mfcc)
    if user_values is None or ref_values is None:
        return None
    try:
        return float(np.linalg.norm(np.array(user_values) - np.array(ref_values)))
    except Exception:
        return None


def build_reference_comparison(
    user_features: FeatureDict,
    reference: ReferenceVector,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for user_key, label, unit, reference_key in _COMPARISON_METRICS:
        user_value = _safe_float(user_features.get(user_key))
        ref_value = _safe_float(reference.get(reference_key))
        diff = None if user_value is None or ref_value is None else user_value - ref_value
        ratio = None
        if user_value is not None and ref_value not in (None, 0):
            ratio = user_value / float(ref_value)
        rows.append(
            {
                "key": user_key,
                "label": label,
                "unit": unit,
                "user_value": user_value,
                "reference_value": ref_value,
                "difference": diff,
                "ratio": ratio,
                "user_width": _percent_width(user_value, ref_value),
                "reference_width": _percent_width(ref_value, user_value),
            }
        )

    mfcc_distance = compute_mfcc_distance(user_features.get("mfcc_mean"), reference.get("mfcc_mean"))
    if mfcc_distance is not None:
        rows.append(
            {
                "key": "mfcc_distance",
                "label": "MFCC distance",
                "unit": "",
                "user_value": mfcc_distance,
                "reference_value": 0.0,
                "difference": mfcc_distance,
                "ratio": None,
                "user_width": min(100.0, mfcc_distance),
                "reference_width": 0.0,
            }
        )
    return rows


def _render_pair_bar(row: dict[str, Any]) -> str:
    user_width = row["user_width"]
    reference_width = row["reference_width"]
    unit = row["unit"]
    user_value = _fmt(row["user_value"], unit)
    reference_value = _fmt(row["reference_value"], unit)
    diff = _fmt(row["difference"], unit)
    ratio = row.get("ratio")
    ratio_text = "-" if ratio is None else f"{ratio:.2f}x"
    return f"""
    <div style="border-top:1px solid #e2e8f0;padding:13px 0;">
      <div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline;margin-bottom:8px;">
        <div style="font-weight:800;color:#334155;">{row['label']}</div>
        <div style="font-size:12px;color:#64748b;">차이 {diff} · 비율 {ratio_text}</div>
      </div>
      <div style="display:grid;grid-template-columns:72px 1fr 74px;gap:8px;align-items:center;margin-bottom:6px;">
        <div style="font-size:12px;font-weight:700;color:#2563eb;text-align:right;">내 음성</div>
        <div style="height:9px;background:#e2e8f0;border-radius:999px;overflow:hidden;"><div style="width:{user_width:.1f}%;height:100%;background:linear-gradient(90deg,#3b82f6,#6366f1);border-radius:999px;"></div></div>
        <div style="font-size:12px;font-weight:800;color:#1e40af;text-align:right;">{user_value}</div>
      </div>
      <div style="display:grid;grid-template-columns:72px 1fr 74px;gap:8px;align-items:center;">
        <div style="font-size:12px;font-weight:700;color:#64748b;text-align:right;">Reference</div>
        <div style="height:9px;background:#e2e8f0;border-radius:999px;overflow:hidden;"><div style="width:{reference_width:.1f}%;height:100%;background:#94a3b8;border-radius:999px;"></div></div>
        <div style="font-size:12px;font-weight:800;color:#475569;text-align:right;">{reference_value}</div>
      </div>
    </div>
    """


def render_reference_comparison_html(
    user_features: FeatureDict,
    reference: ReferenceVector,
    *,
    title: str = "Reference Comparison",
) -> str:
    rows = build_reference_comparison(user_features, reference)
    if not rows:
        return ""
    rendered_rows = "".join(_render_pair_bar(row) for row in rows)
    return f"""
    <div class="metrics-card">
      <div class="card-lbl">{title}</div>
      <div class="card-body" style="font-size:13px;color:#64748b;margin-bottom:4px;">내 음성과 영어 reference의 주요 음향 feature를 비교합니다.</div>
      {rendered_rows}
    </div>
    """


def render_onset_analysis_html(details: FeatureDict) -> str:
    if "onset_window_ms" not in details:
        return ""
    onset_rows = [
        ("onset_window_ms", "Window", "ms"),
        ("onset_zcr_mean", "Onset ZCR", ""),
        ("onset_rms_mean", "Onset RMS", ""),
        ("onset_spectral_centroid_mean", "Onset spectral", "Hz"),
    ]
    items: list[str] = []
    for key, label, unit in onset_rows:
        value = _safe_float(details.get(key))
        if value is None:
            continue
        items.append(
            f"<div style='display:flex;justify-content:space-between;border-top:1px solid #e2e8f0;padding:9px 0;'>"
            f"<span style='font-weight:700;color:#334155;'>{label}</span>"
            f"<span style='font-weight:800;color:#1e40af;'>{_fmt(value, unit)}</span>"
            f"</div>"
        )
    if not items:
        return ""
    target_position = details.get("target_position", "onset")
    return f"""
    <div class="metrics-card">
      <div class="card-lbl">Onset Analysis</div>
      <div class="card-body" style="font-size:13px;color:#64748b;margin-bottom:4px;">타겟 위치: {target_position} · 앞 200ms 구간의 사용자 음성 feature입니다.</div>
      {''.join(items)}
    </div>
    """


def _parse_details_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def list_recording_choices(db_path: str | Path = DEFAULT_DB_PATH, limit: int = 100) -> list[tuple[str, int]]:
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, word, phoneme, score, test_label
            FROM user_recordings
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    choices: list[tuple[str, int]] = []
    for row in rows:
        score = row["score"]
        score_text = "--" if score is None else f"{float(score):.1f}"
        label = row["test_label"] or "app"
        choices.append((f"#{row['id']} · {row['word']} /{row['phoneme']}/ · {label} · {score_text}", int(row["id"])))
    return choices


def render_saved_recording_analysis(
    recording_id: int | None,
    reference_vectors: dict[str, ReferenceVector],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    if recording_id is None:
        return "<div class='metrics-card'><div class='card-lbl'>Analysis</div><div class='card-body'>분석할 녹음을 선택해주세요.</div></div>"
    db_path = Path(db_path)
    if not db_path.exists():
        return "<div class='metrics-card'><div class='card-lbl'>Analysis</div><div class='card-body'>저장된 녹음 DB가 없습니다.</div></div>"
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT *
            FROM user_recordings
            WHERE id = ?
            """,
            (recording_id,),
        ).fetchone()
    if row is None:
        return "<div class='metrics-card'><div class='card-lbl'>Analysis</div><div class='card-body'>선택한 녹음을 찾을 수 없습니다.</div></div>"

    phoneme = row["phoneme"]
    reference = reference_vectors.get(phoneme)
    if reference is None:
        return "<div class='metrics-card'><div class='card-lbl'>Analysis</div><div class='card-body'>해당 음소의 reference가 없습니다.</div></div>"

    details = _parse_details_json(row["details_json"])
    user_features: FeatureDict = {
        "duration_ms": row["duration_ms"],
        "zcr_mean": row["zcr_mean"],
        "rms_mean": row["rms_mean"],
        "spectral_centroid_mean": row["spectral_centroid_mean"],
    }
    if row["mfcc_distance"] is not None:
        user_features["mfcc_distance"] = row["mfcc_distance"]
    user_features.update(details)

    score = row["score"]
    score_text = "--" if score is None else f"{float(score):.1f}"
    header = f"""
    <div class="target-card">
      <div class="target-word">{row['word']}</div>
      <div class="target-phoneme">/{phoneme}/ · score {score_text}</div>
      <div class="target-hint">저장 ID #{row['id']} · {row['created_at']}</div>
    </div>
    """
    return (
        header
        + render_reference_comparison_html(user_features, reference, title="Saved Reference Comparison")
        + render_onset_analysis_html(details)
    )
