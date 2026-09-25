# P3-v2 post-primary result archive

The author supplied `p3_v2_analysis_v1.zip` on 2026-09-25. Its only two
members were copied here without editing. Their SHA-256 digests are:

- `p3_v2_analysis_report.json`:
  `374af7297a4fb24e712ef1457ebce6a0ac94f57767849ec60390a41491e43813`
- `complete_twelve_cell_matrix.csv`:
  `210e9b1a15a6ffd58477a2a272a1bae19b1b782f3cc14bff317186f84756531f`

The report's embedded matrix digest matches the CSV. It identifies the frozen
P3-v2 config (`2d6226ee0155cd9decc8c95a6643f1fa93003c92b087601ea448b61eca30960e`),
the model-runner code digest
(`c31de65097adeccd962ed587ef32d5715b8981735f485ea8160765c44940f68e`),
and the analysis-code digest
(`3437e70884319a228305f0a91db92460e4bfeb3e29a4e944dacfe5afe3e44791`).
Both code digests matched the repository when these files were imported.

The verified aggregate contains all 12 backbone-by-source-by-mechanism cells,
571 valid scenarios per backbone after five construct exclusions, and 32
original source-ID clusters in each cell. Three complete cells pass the
unchanged P1 cell rule descriptively, but all are on `traffic_daily`. The
registered two-source transfer condition therefore **fails**. P3-v1's failed
construct pilot remains disclosed separately; neither P3 result changes P1.

Lower-link relative SQL is unavailable because lower-link quantiles were not
archived. The CSV contains no original source IDs, background windows, or
forecast arrays. The report records hashes for 74 private completion and unit
artifacts, but those private artifacts were not supplied in this ZIP and were
not independently re-audited during this import. This archive supports only
the bounded post-primary claims stated above.
