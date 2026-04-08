param(
    [Parameter(Mandatory = $true)]
    [string]$SpaceId,

    [string]$Branch = "main",
    [string]$RemoteName = "space",
    [string]$CommitMessage = "Deploy Adaptive Learning System to Hugging Face Spaces"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

Push-Location $repoRoot
try {
    if (-not (Test-Path ".git")) {
        git init | Out-Null
    }

    $remoteUrl = "https://huggingface.co/spaces/$SpaceId"
    $existingRemoteUrl = ""

    try {
        $existingRemoteUrl = (git remote get-url $RemoteName).Trim()
    } catch {
        $existingRemoteUrl = ""
    }

    if ($existingRemoteUrl) {
        git remote set-url $RemoteName $remoteUrl
    } else {
        git remote add $RemoteName $remoteUrl
    }

    git add .
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m $CommitMessage
    }

    git push --set-upstream $RemoteName HEAD:$Branch
} finally {
    Pop-Location
}
