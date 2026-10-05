# Run after run_burst_benchmark.py and train_burst_wait.py complete.
Set-Location $PSScriptRoot\..

Write-Host "Summarizing logs..."
.\.venv\Scripts\python.exe scripts\summarize_burst_results.py

Write-Host "Plot (PNG if matplotlib available, else CSV)..."
.\.venv\Scripts\python.exe scripts\plot_burst_benchmark.py

Write-Host "Done. See docs/EXPERIMENTS_COMPLETE.md and outputs/logs/BENCHMARK_SUMMARY.json"
