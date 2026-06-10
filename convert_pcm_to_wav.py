import os
import numpy as np
import soundfile as sf
from pathlib import Path

PCM_DIR = Path("data/pcm_original")
WAV_DIR = Path("data/wav_converted")
SAMPLERATE = 16000
DTYPE = np.int16

WAV_DIR.mkdir(exist_ok=True)

pcm_files = sorted(PCM_DIR.glob("*.pcm"))

if not pcm_files:
    print("pcm_original/ 에 .pcm 파일이 없습니다.")
    exit(0)

converted = []
failed = []

for pcm_path in pcm_files:
    wav_path = WAV_DIR / pcm_path.with_suffix(".wav").name
    try:
        raw = np.frombuffer(pcm_path.read_bytes(), dtype=DTYPE)
        sf.write(str(wav_path), raw, SAMPLERATE, subtype="PCM_16")
        converted.append(wav_path)
        print(f"[OK] {pcm_path.name} -> {wav_path.name}")
    except Exception as e:
        failed.append((pcm_path.name, str(e)))
        print(f"[FAIL] {pcm_path.name}: {e}")

print(f"\n변환 완료: {len(converted)}개 / 실패: {len(failed)}개")

if failed:
    print("\n실패 목록:")
    for name, reason in failed:
        print(f"  - {name}: {reason}")

if converted:
    first_wav = str(converted[0])
    print(f"\n첫 번째 파일 재생: {converted[0].name}")
    try:
        import sounddevice as sd
        data, sr = sf.read(first_wav, dtype="int16")
        sd.play(data.astype(np.float32) / 32768.0, sr)
        sd.wait()
    except Exception:
        os.startfile(first_wav)
