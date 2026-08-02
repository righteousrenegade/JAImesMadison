from pathlib import Path
import json
import re

SOURCE = Path(__file__).parents[1] / "federalistpapers.txt"
OUTPUT_DIR = Path(__file__).parent / "federalist_papers"

MARKER_RE = re.compile(r"^[ \t]*FEDERALIST No\.\s*(\d+)\..*$", re.MULTILINE)
LIGATURE_REPLACEMENTS = {
    "\u017f": "s",  # long s
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}


def normalize_ocr(text: str) -> str:
    for source, replacement in LIGATURE_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    return text


def main() -> None:
    source_text = SOURCE.read_text(encoding="utf-8")
    markers = list(MARKER_RE.finditer(source_text))
    expected_numbers = list(range(1, 86))
    actual_numbers = [int(marker.group(1)) for marker in markers]

    if actual_numbers != expected_numbers:
        raise ValueError(
            f"Expected Federalist markers 1-85, found {actual_numbers}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in OUTPUT_DIR.glob("*.txt"):
        old_file.unlink()

    manifest = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(source_text)
        document = normalize_ocr(source_text[marker.start():end]).strip() + "\n"
        number = int(marker.group(1))
        filename = f"{number:03d}_federalist_{number}.txt"
        (OUTPUT_DIR / filename).write_text(document, encoding="utf-8", newline="\n")
        manifest.append(
            {
                "number": number,
                "filename": filename,
                "title": marker.group(0).strip(),
                "characters": len(document),
            }
        )

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(SOURCE.name),
                "document_count": len(manifest),
                "encoding": "UTF-8",
                "normalization": "Replace long s and common Unicode ligatures; preserve capitalization and spelling.",
                "documents": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
