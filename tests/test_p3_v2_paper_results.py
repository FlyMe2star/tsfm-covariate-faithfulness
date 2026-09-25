"""Guard the imported post-primary aggregate and its manuscript table."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/p3_semisynthetic/v2_results"
REPORT = EVIDENCE / "p3_v2_analysis_report.json"
MATRIX = EVIDENCE / "complete_twelve_cell_matrix.csv"
TABLE = ROOT / "paper/results/p3_v2_transfer_table.tex"


def test_imported_aggregate_is_complete_and_bounded() -> None:
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == (
        "374af7297a4fb24e712ef1457ebce6a0ac94f57767849ec60390a41491e43813"
    )
    matrix_hash = hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    assert matrix_hash == "210e9b1a15a6ffd58477a2a272a1bae19b1b782f3cc14bff317186f84756531f"
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    with MATRIX.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert report["complete_matrix_sha256"] == matrix_hash
    assert report["result_status"] == "verified_post_primary_descriptive_analysis"
    assert report["counts"]["valid_scenarios_per_backbone"] == 571
    assert len(rows) == len(report["cells"]) == 12
    assert sum(row["complete_cell"] == "True" for row in rows) == 3
    assert {row["dataset"] for row in rows if row["complete_cell"] == "True"} == {
        "traffic_daily"
    }
    assert report["decision"]["bounded_transfer_supported"] is False
    assert report["decision"]["lower_link_sql_missing"] is True
    assert all(row["lower_relative_sql"] == "" for row in rows)
    assert all(cell["lower_link_relative_sql"] is None for cell in report["cells"])
    assert not any("source_id" in row for row in rows)
    by_key = {
        (cell["backbone"], cell["dataset"], cell["family"]): cell
        for cell in report["cells"]
    }
    for row in rows:
        cell = by_key[(row["backbone"], row["dataset"], row["family"])]
        assert int(row["n"]) == cell["valid_scenario_count"]
        assert int(row["source_clusters"]) == cell["source_id_cluster_count"]
        assert (row["complete_cell"] == "True") is cell["complete_cell"]
        for field, group, metric, member in (
            ("dsa", "primary", "dsa", "estimate"),
            ("dsa_lower", "primary", "dsa", "lower"),
            ("rgr", "primary", "rgr", "estimate"),
            ("rgr_lower", "primary", "rgr", "lower"),
            ("rgr_upper", "primary", "rgr", "upper"),
            ("d1", "primary", "d1", "estimate"),
            ("d1_lower", "primary", "d1", "lower"),
            ("g", "primary", "g", "estimate"),
            ("g_lower", "primary", "g", "lower"),
            ("sham_ratio", "diagnostic_and_controls", "sham_ratio", "estimate"),
        ):
            assert float(row[field]) == cell[group][metric][member]
        assert float(row["relative_sql"]) == cell["relative_sql"]
        assert float(row["relative_wql"]) == cell["relative_wql"]


def test_every_displayed_p3_v2_table_row_matches_unrounded_csv() -> None:
    with MATRIX.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    table_lines = [
        line.strip()
        for line in TABLE.read_text(encoding="utf-8").splitlines()
        if line.startswith(("Chronos-2 &", "TimesFM-3 &"))
    ]
    assert len(table_lines) == len(rows) == 12
    model = {"chronos_2": "Chronos-2", "timesfm_3": "TimesFM-3"}
    dataset = {
        "traffic_daily": "Traffic",
        "workload_5min": "Workload",
        "solar_10min": "Solar",
    }
    family = {"biphasic_rebound": "Biphasic", "dispersed_delayed_pulse": "Delayed"}
    for line, row in zip(table_lines, rows, strict=True):
        assert line.endswith(r"\\")
        fields = [field.strip() for field in line[:-2].split("&")]
        g_lower = float(row["g_lower"])
        expected = [
            model[row["backbone"]],
            dataset[row["dataset"]],
            family[row["family"]],
            f"{float(row['dsa_lower']):.3f}",
            f"[{float(row['rgr_lower']):.3f}, {float(row['rgr_upper']):.3f}]",
            f"{float(row['d1_lower']):.3f}",
            f"{g_lower:.4f}" if g_lower < 0.001 else f"{g_lower:.3f}",
            f"${float(row['relative_sql']) * 100:+.1f}$",
            "Yes" if row["complete_cell"] == "True" else "No",
        ]
        assert fields == expected
