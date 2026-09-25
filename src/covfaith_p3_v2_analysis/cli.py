"""Fresh-process entry point for the read-only P3-v2 archive analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from covfaith_p3.data import sha256_file

from .analysis import analyze_full_archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = analyze_full_archives(args.repo, args.data_root, args.archive_root)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"P3_V2_ANALYSIS_ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    report_path = args.archive_root / "analysis_v1/p3_v2_analysis_report.json"
    matrix_path = args.archive_root / "analysis_v1/complete_twelve_cell_matrix.csv"
    print("Result status:", report["result_status"])
    print("Report SHA-256:", sha256_file(report_path))
    print("Matrix SHA-256:", sha256_file(matrix_path))
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    print("All twelve cells:")
    print(matrix_path.read_text(encoding="utf-8"))
    print("Private report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
