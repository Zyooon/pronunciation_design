import time
from pathlib import Path

from gtts import gTTS

WORDS_FILE = Path("data/words.txt")
EN_DIR = Path("data/reference_en/gtts")
KO_DIR = Path("data/reference_ko/gtts")
DELAY = 0.5


def parse_words(path: Path) -> list[tuple[str, str]]:
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        words.append((parts[0].strip(), parts[1].strip()))
    return words


def generate(text: str, lang: str, tld: str | None, out_path: Path) -> str:
    if out_path.exists():
        return "SKIP"
    try:
        kwargs = {"lang": lang}
        if tld:
            kwargs["tld"] = tld
        gTTS(text=text, **kwargs).save(str(out_path))
        return "OK"
    except Exception as e:
        return f"FAIL ({e})"


def main() -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    KO_DIR.mkdir(parents=True, exist_ok=True)

    words = parse_words(WORDS_FILE)

    counts = {"OK": 0, "SKIP": 0, "FAIL": 0}

    for en_word, ko_word in words:
        # English
        en_path = EN_DIR / f"{en_word}.mp3"
        en_status = generate(en_word, "en", "us", en_path)
        label = "SKIP (already exists)" if en_status == "SKIP" else en_status
        print(f"[EN] {en_path.name} → {label}")
        counts[en_status if en_status in counts else "FAIL"] += 1
        if en_status == "OK":
            time.sleep(DELAY)

        # Korean
        ko_path = KO_DIR / f"{en_word}_ko.mp3"
        ko_status = generate(ko_word, "ko", None, ko_path)
        label = "SKIP (already exists)" if ko_status == "SKIP" else ko_status
        print(f"[KO] {ko_path.name} → {label}")
        counts[ko_status if ko_status in counts else "FAIL"] += 1
        if ko_status == "OK":
            time.sleep(DELAY)

    print(f"\n완료 — 성공: {counts['OK']}  스킵: {counts['SKIP']}  실패: {counts['FAIL']}")


if __name__ == "__main__":
    main()
