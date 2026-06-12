/**
 * 브라우저 녹음 기능만 담당한다.
 * UI 조작은 하지 않는다. 녹음 완료 시 'recordingcomplete' 이벤트를 발행한다.
 */

let _mediaRecorder = null;
let _chunks        = [];
let _recordedBlob  = null;
let _isRecording   = false;

/**
 * 마이크 권한을 요청하고 녹음을 시작한다.
 * 완료 또는 실패 시 호출자가 이벤트를 통해 감지한다.
 * @throws {DOMException} 마이크 권한 거부 시
 */
async function startRecording() {
  _chunks       = [];
  _recordedBlob = null;

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  const mimeType = _chooseMimeType();
  const options  = mimeType ? { mimeType } : {};
  _mediaRecorder = new MediaRecorder(stream, options);

  _mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) _chunks.push(e.data);
  };

  _mediaRecorder.onstop = () => {
    const blob = new Blob(_chunks, { type: _mediaRecorder.mimeType || "audio/webm" });
    _recordedBlob = blob;
    _isRecording  = false;

    // 스트림 트랙 정리
    stream.getTracks().forEach((t) => t.stop());

    document.dispatchEvent(
      new CustomEvent("recordingcomplete", { detail: { blob } })
    );
  };

  _mediaRecorder.start(200); // 200ms 단위로 chunk 수집
  _isRecording = true;
}

/**
 * 녹음을 중지한다. onstop이 발화된 뒤 'recordingcomplete' 이벤트가 전달된다.
 */
function stopRecording() {
  if (_mediaRecorder && _isRecording) {
    _mediaRecorder.stop();
  }
}

/** 녹음된 Blob을 반환한다. 녹음 전이면 null. */
function getRecordedBlob() {
  return _recordedBlob;
}

/** 내부 상태를 초기화한다. */
function resetRecording() {
  if (_mediaRecorder && _isRecording) {
    _mediaRecorder.stop();
  }
  _mediaRecorder = null;
  _chunks        = [];
  _recordedBlob  = null;
  _isRecording   = false;
}

/** 현재 녹음 중인지 반환한다. */
function isRecording() {
  return _isRecording;
}

/** 브라우저가 지원하는 mimeType을 선택한다. */
function _chooseMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
  ];
  return candidates.find((t) => MediaRecorder.isTypeSupported(t)) || "";
}
