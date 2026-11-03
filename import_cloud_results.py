"""Import the archive downloaded through the Colab UI and verify its provenance."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    target = ROOT / "actual_run"
    target.mkdir(exist_ok=True)
    allowed = {".json", ".jsonl", ".csv", ".png", ".pdf", ".py"}
    with zipfile.ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        for item in archive.infolist():
            p = PurePosixPath(item.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in item.filename:
                raise ValueError("Unsafe archive path")
            if p.parts[0] not in {"task1_assignment", "source"} or p.suffix not in allowed:
                raise ValueError(f"Unexpected archive item: {item.filename}")
            if item.is_dir() or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unexpected directory or symbolic link")
            destination = target.joinpath(*p.parts)
            if not destination.resolve().is_relative_to(target.resolve()):
                raise ValueError("Archive escaped destination")
            raw = archive.read(item)
            if destination.exists() and destination.read_bytes() != raw:
                raise ValueError(f"Refusing to replace different results: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
    for name in ("runner.py", "cloud_driver.py", "analyze_results.py", "data_quality_review.json"):
        assert (target / "source" / name).read_bytes() == (ROOT / name).read_bytes(), name
    from analyze_results import load_run
    formal = target / "task1_assignment/formal"
    load_run(formal, formal / "source_data.json")
    receipt = {"archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
               "source_bytes_match": True, "formal_log_validation": "passed",
               "archive_bytes": args.archive.stat().st_size}
    (target / "import_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print(formal.parent)


if __name__ == "__main__":
    main()
