/**
 * API 호출만 담당한다.
 * UI 조작이나 상태 관리는 하지 않는다.
 */

async function fetchWords() {
  try {
    const response = await fetch("/api/words");
    if (!response.ok) {
      throw new Error(`단어 목록 로드 실패 (${response.status})`);
    }
    return response.json();
  } catch (err) {
    if (err instanceof TypeError) {
      throw new Error("서버에 연결할 수 없습니다. 네트워크 상태를 확인해주세요.");
    }
    throw err;
  }
}

async function analyzePronunciation(formData) {
  try {
    const response = await fetch("/api/pronunciation/analyze", {
      method: "POST",
      body: formData,
    });

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(`서버 응답을 읽을 수 없습니다 (${response.status})`);
    }

    if (!response.ok) {
      const detail = data.detail;
      // FastAPI 검증 오류(list) vs 커스텀 오류(string)
      const message = Array.isArray(detail)
        ? detail.map((d) => d.msg || d.message || JSON.stringify(d)).join(", ")
        : (detail || `분석 실패 (${response.status})`);
      throw new Error(message);
    }

    return data;
  } catch (err) {
    if (err instanceof TypeError) {
      throw new Error("서버에 연결할 수 없습니다. 네트워크 상태를 확인해주세요.");
    }
    throw err;
  }
}
