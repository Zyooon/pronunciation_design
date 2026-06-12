import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent / "scripts"

SCRIPTS: dict[str, dict] = {
    "1": {
        "name": "generate_reference_audio.py",
        "description": "ElevenLabs API로 영어 레퍼런스 오디오 생성",
        "ask_limit": True,
    },
    "2": {
        "name": "build_gtts.py",
        "description": "gTTS로 영어/한국어 오디오 생성",
        "ask_limit": False,
    },
    "3": {
        "name": "build_reference.py",
        "description": "레퍼런스 벡터 빌드 (reference_vectors.json)",
        "ask_limit": False,
    },
    "4": {
        "name": "compare_en_ko.py",
        "description": "영어-한국어식 발음 비교 및 DB/JSON 저장",
        "ask_limit": False,
    },
}


def print_menu() -> None:
    print("\n============================")
    print("   Pronunciation Design")
    print("============================")
    print("실행할 스크립트를 선택하세요:\n")
    for key, info in SCRIPTS.items():
        print(f"  {key}. {info['name']}")
        print(f"     └─ {info['description']}")
    print("\n  0. 종료")
    print()


def ask_limit() -> int | None:
    print("\n최대 몇 개의 단어(row)를 처리할까요?")
    print("  전체를 처리하려면 그냥 Enter를 누르세요.")
    raw = input("개수 입력: ").strip()
    if not raw:
        return None
    try:
        n = int(raw)
        if n <= 0:
            print("[오류] 1 이상의 숫자를 입력해 주세요.")
            return ask_limit()
        return n
    except ValueError:
        print("[오류] 숫자를 입력해 주세요.")
        return ask_limit()


def ask_confirm(script_name: str, limit: int | None = None) -> bool:
    parts = [f"'{script_name}'"]
    if limit is not None:
        parts.append(f"(최대 {limit}개 처리)")
    prompt = " ".join(parts) + " 을(를) 실행하시겠습니까? [y/N]: "
    answer = input(prompt).strip().lower()
    return answer in ("y", "yes", "예", "ㅇ")


def run_script(script_name: str, limit: int | None = None) -> None:
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    subprocess.run(cmd, cwd=Path(__file__).parent)


def main() -> None:
    while True:
        print_menu()
        choice = input("선택 (번호 입력): ").strip()

        if choice == "0":
            print("종료합니다.")
            break

        if choice not in SCRIPTS:
            print("[오류] 올바른 번호를 입력해 주세요.")
            continue

        script_info = SCRIPTS[choice]
        script_name = script_info["name"]
        limit: int | None = None

        if script_info["ask_limit"]:
            limit = ask_limit()

        if not ask_confirm(script_name, limit):
            print("취소되었습니다.")
            continue

        run_script(script_name, limit)


if __name__ == "__main__":
    main()
