/**
 * API 호출만 담당한다.
 * UI 조작이나 상태 관리는 하지 않는다.
 */

async function fetchWords() {
  const response = await fetch("/api/words");
  if (!response.ok) {
    throw new Error(`단어 목록 로드 실패: ${response.status}`);
  }
  return response.json();
}

async function analyzePronunciation(formData) {
  const response = await fetch("/api/pronunciation/analyze", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `분석 실패: ${response.status}`);
  }
  return data;
}
