/**
 * 브라우저 녹음 기능만 담당한다.
 * UI 조작은 하지 않는다. 녹음 완료 시 'recordingcomplete' 이벤트를 발행한다.
 * 녹음 데이터는 서버 호환성을 위해 WAV(16-bit PCM, 16kHz mono)로 변환한다.
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

  _mediaRecorder.onstop = async () => {
    const rawBlob = new Blob(_chunks, { type: _mediaRecorder.mimeType || "audio/webm" });
    _isRecording  = false;

    stream.getTracks().forEach((t) => t.stop());

    try {
      _recordedBlob = await _convertToWav(rawBlob);
    } catch (err) {
      console.warn("WAV 변환 실패, 원본 형식으로 전송합니다:", err);
      _recordedBlob = rawBlob;
    }

    document.dispatchEvent(
      new CustomEvent("recordingcomplete", { detail: { blob: _recordedBlob } })
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

/**
 * Blob을 WAV(16-bit PCM, 16kHz mono)로 변환한다.
 * Web Audio API의 OfflineAudioContext로 디코딩 및 리샘플링한다.
 */
async function _convertToWav(blob) {
  const TARGET_SR = 16000;

  const arrayBuffer = await blob.arrayBuffer();

  const decodeCtx = new AudioContext();
  const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
  await decodeCtx.close();

  const offlineCtx = new OfflineAudioContext(
    1,
    Math.ceil(audioBuffer.duration * TARGET_SR),
    TARGET_SR,
  );
  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineCtx.destination);
  source.start();

  const resampled = await offlineCtx.startRendering();
  const pcm = resampled.getChannelData(0);

  return new Blob([_encodeWav(pcm, TARGET_SR)], { type: "audio/wav" });
}

/** Float32 PCM 샘플 배열을 WAV ArrayBuffer로 인코딩한다. */
function _encodeWav(samples, sampleRate) {
  const bitsPerSample = 16;
  const numChannels   = 1;
  const dataLength    = samples.length * (bitsPerSample / 8);
  const buffer        = new ArrayBuffer(44 + dataLength);
  const view          = new DataView(buffer);

  _writeAscii(view, 0,  "RIFF");
  view.setUint32(4,  36 + dataLength, true);
  _writeAscii(view, 8,  "WAVE");
  _writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);                                          // chunk size
  view.setUint16(20, 1,  true);                                          // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * (bitsPerSample / 8), true);
  view.setUint16(32, numChannels * (bitsPerSample / 8), true);
  view.setUint16(34, bitsPerSample, true);
  _writeAscii(view, 36, "data");
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF, true);
    offset += 2;
  }

  return buffer;
}

function _writeAscii(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}
