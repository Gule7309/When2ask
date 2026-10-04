param(
    [int]$Limit = 20,
    [int]$Seed = 0,
    [double]$TauExec = 0.8,
    [int]$NCandidates = 5
)

$ErrorActionPreference = "Stop"

$Model = "qwen2.5:7b"
$BaseUrl = "http://localhost:11434/v1"
$Output = "results/qwen2.5-7b/initial_probe_n${NCandidates}_tau${TauExec}_seed${Seed}.jsonl"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama is not installed or is not on PATH."
}

if (-not (Test-Path "third_party/ClarifyBench")) {
    python scripts/bootstrap_clarifybench.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

ollama pull $Model
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$env:OPENAI_BASE_URL = $BaseUrl
$env:OPENAI_API_KEY = "ollama"

python scripts/run_initial_candidate_probe.py --model $Model --output $Output --tau-exec $TauExec --n-candidates $NCandidates --temperature 0.5 --limit $Limit --seed $Seed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Probe written to $Output"
Write-Host "Next: python scripts/summarize_initial_probe.py $Output"
