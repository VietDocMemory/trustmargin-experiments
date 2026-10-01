$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$rag = (Resolve-Path (Join-Path $root '..\vietnamese-rag-system')).Path

function Test-LocalPort([int]$port) {
    return [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

if (-not (Test-LocalPort 6333)) {
    $storage = (New-Item -ItemType Directory -Path (Join-Path $rag 'qdrant_storage\v1.17.1') -Force).FullName
    Start-Process -FilePath (Join-Path $env:LOCALAPPDATA 'qdrant-1.17.1\qdrant.exe') -WorkingDirectory $storage -WindowStyle Hidden
}
if (-not (Test-LocalPort 6379)) {
    Start-Service Memurai
}
if (-not (Test-LocalPort 11434)) {
    Start-Process -FilePath (Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe') -ArgumentList serve -WindowStyle Hidden
}
if (-not (Test-LocalPort 8000)) {
    Start-Process -FilePath (Join-Path $rag '.venv\Scripts\python.exe') -ArgumentList '-m uvicorn src.api.main:app --host 127.0.0.1 --port 8000' -WorkingDirectory $rag -WindowStyle Hidden
}

Write-Output 'RAG services started or already listening on ports 6333, 6379, 11434, and 8000.'
