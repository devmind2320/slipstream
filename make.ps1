# Windows shim for the Makefile targets — `make` is not present on a stock
# Windows install and the spec's entry points have to work here too.
#
#   .\make.ps1 demo
#   .\make.ps1 eval -Scale 0.25
#
# Every target below is the same command the Makefile runs.

param(
    [Parameter(Position = 0)][string]$Target = "help",
    [int]$Seed = 1729,
    [double]$Scale = 1.0,
    [double]$Alpha = 0.10,
    [int]$Concurrency = 24,
    [int]$Port = 8000
)

$py = "py"
$pyArgs = @("-3.11")
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { $py = "python"; $pyArgs = @() }

function Run([string[]]$rest) {
    & $py @($pyArgs + $rest)
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target) {
    "help" {
        Write-Host "Slipstream - adaptive verification scheduling"
        Write-Host ""
        Write-Host "  .\make.ps1 install     install pinned dependencies"
        Write-Host "  .\make.ps1 demo        offline A/B demo - no network, no API key"
        Write-Host "  .\make.ps1 serve       gateway + dashboard on http://127.0.0.1:$Port"
        Write-Host "  .\make.ps1 eval        regenerate the full three-arm results table"
        Write-Host "  .\make.ps1 eval-fast   the same at 25% corpus scale"
        Write-Host "  .\make.ps1 test        full test suite"
        Write-Host "  .\make.ps1 verify      walk the ledger hash chain"
    }
    "install"        { Run @("-m", "pip", "install", "-r", "requirements.txt") }
    "demo"           { Run @("-m", "slipstream", "demo", "--scenario", "compensation") }
    "demo-injection" { Run @("-m", "slipstream", "demo", "--scenario", "injection") }
    "demo-benign"    { Run @("-m", "slipstream", "demo", "--scenario", "benign") }
    "serve" {
        Write-Host "dashboard -> http://127.0.0.1:$Port/"
        Run @("-m", "slipstream", "serve", "--port", "$Port")
    }
    "data"  { Run @("-m", "slipstream", "data", "--seed", "$Seed", "--scale", "$Scale") }
    "eval"  { Run @("-m", "slipstream", "eval", "--seed", "$Seed", "--scale", "$Scale",
                    "--alpha", "$Alpha", "--concurrency", "$Concurrency") }
    "eval-fast" { Run @("-m", "slipstream", "eval", "--seed", "$Seed", "--scale", "0.25",
                    "--alpha", "$Alpha", "--concurrency", "$Concurrency") }
    "eval-deterministic" { Run @("-m", "slipstream.eval.harness", "--seed", "$Seed",
                    "--scale", "$Scale", "--alpha", "$Alpha",
                    "--concurrency", "$Concurrency", "--deterministic") }
    "report" { Get-Content "slipstream/eval/results/report.md" }
    "fit"    { Run @("-m", "slipstream", "fit", "--alpha", "$Alpha", "--seed", "$Seed", "--report") }
    "recalibrate" { Run @("-m", "slipstream", "recalibrate", "--alpha", "$Alpha", "--seed", "$Seed") }
    "test"   { Run @("-m", "pytest", "tests/") }
    "verify" { Run @("-m", "slipstream", "ledger", "verify") }
    "info"   { Run @("-m", "slipstream", "info") }
    "clean" {
        Remove-Item -Recurse -Force data, .pytest_cache -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Directory -Filter __pycache__ |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
    default { Write-Host "unknown target '$Target' - try: .\make.ps1 help"; exit 1 }
}
