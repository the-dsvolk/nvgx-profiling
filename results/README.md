# Results

One Markdown file per run: `YYYY-MM-DD-<option>-<short-name>.md`.

Record: box (DGX Spark #1/#2), option (A/B), kernel, driver (kernel-mode and user-mode), CUDA toolkit,
SDK version, workload, what passed/failed against the [gates](../plan/README.md#gates), overhead numbers,
and screenshots of interesting Perfetto views.

Raw traces (`.nvtsf`, `.perfetto-trace.zip`, DuckDB) are not committed (see `.gitignore`).
Strip host names, user names and paths from anything copied in here.
