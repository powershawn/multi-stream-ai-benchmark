param(
    [ValidateSet('run', 'smoke')]
    [string]$Mode = 'run',
    [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath './video.mp4' -PathType Leaf)) {
        throw 'video.mp4 is missing from the project folder.'
    }
    docker info --format '{{.ServerVersion}}'
    if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop with the WSL 2 Linux engine first.' }
    if (-not $NoBuild) {
        docker compose build
        if ($LASTEXITCODE -ne 0) { throw 'Docker build failed.' }
    }
    docker compose run --rm benchmark prepare
    if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed.' }
    # Collect after slow setup, pass to only this invocation; no reusable host file.
    $hostSnapshot = & (Join-Path $PSScriptRoot 'scripts/Get-HostInfo.ps1') -AsBase64
    if (-not $hostSnapshot) { throw 'Host snapshot collector returned no data.' }
    if ($Mode -eq 'smoke') {
        docker compose run --rm -e "BENCHMARK_HOST_INFO_B64=$hostSnapshot" benchmark smoke --duration 15
    } else {
        docker compose run --rm -e "BENCHMARK_HOST_INFO_B64=$hostSnapshot" benchmark run
    }
    $benchmarkExit = $LASTEXITCODE
    if ($benchmarkExit -eq 1) {
        Write-Host 'Acceptance criteria were not met. See results/ for measurements and failure reasons.'
    } elseif ($benchmarkExit -ne 0) {
        throw "Benchmark failed with exit code $benchmarkExit. See results/ for diagnostics."
    }
} finally {
    Pop-Location
}
exit $benchmarkExit
