"""ElevenLabs API를 사용해 영어 reference 오디오를 생성하는 스크립트.

words.txt를 읽어 여러 화자(Bella, Sarah, George 등)의 음성을 생성하고
data/reference_en/<Voice>/단어.mp3 경로로 저장한다. ELEVENLABS_API_KEY 필요.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs

load_dotenv()

WORDS_FILE = Path("data/words.txt")
EN_DIR = Path("data/reference_en")

# ElevenLabs 프리셋 목소리 ID
VOICES: dict[str, str] = {
    "Bella":   "hpp4J3VqNfWAUOO0d1Us",
    "Sarah":   "EXAVITQu4vr4xnSDxMaL",
    "Jessica": "cgSgspJ2msm6clMCkdW9",
    "Matilda": "XrExE9yKIg1WjnnlVkGX",
    "Brian":   "nPczCjzI2devNBz1zQrb",
    "George":  "JBFqnCBsd6RMkjVDRZzb",
    "Adam":    "pNInz6obpgDQGcFmaJgB",
    "River":   "SAz9YHcvj6GT2YYXdXww",
}


def load_words(path: Path) -> list[tuple[str, str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            print(f"[SKIP] 형식 오류: {line!r}")
            continue
        word, ko_pron, phoneme = parts[0].strip(), parts[1].strip(), parts[2].strip()
        entries.append((word, ko_pron, phoneme))
    return entries


def generate_elevenlabs(client: ElevenLabs, text: str, voice_id: str, dest: Path) -> None:
    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    with open(dest, "wb") as f:
        for chunk in audio:
            f.write(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ElevenLabs API로 영어 레퍼런스 오디오 파일을 생성합니다."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 최대 단어(row) 수. 미지정 시 전체 처리.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        raise SystemExit("[ERROR] .env 파일에 유효한 ELEVENLABS_API_KEY를 설정해 주세요.")

    client = ElevenLabs(api_key=api_key)

    for voice_name in VOICES:
        (EN_DIR / voice_name).mkdir(parents=True, exist_ok=True)

    words = load_words(WORDS_FILE)
    if args.limit is not None:
        words = words[: args.limit]
    print(f"단어 {len(words)}개 로드 완료\n")

    success: list[str] = []
    failed: list[str] = []

    for word, ko_pron, phoneme in words:
        for voice_name, voice_id in VOICES.items():
            dest = EN_DIR / voice_name / f"{word}.mp3"
            try:
                generate_elevenlabs(client, word, voice_id, dest)
                success.append(str(dest))
                print(f"[OK]  {dest}")
            except Exception as e:
                failed.append(str(dest))
                print(f"[FAIL] {dest}: {e}")

    print("\n=== 결과 ===")
    print(f"성공 ({len(success)}개):")
    for p in success:
        print(f"  [OK]  {p}")

    if failed:
        print(f"\n실패 ({len(failed)}개):")
        for p in failed:
            print(f"  [ERR] {p}")

    print(f"\n총 생성 파일: {len(success)} / {len(success) + len(failed)}")


if __name__ == "__main__":
    main()
