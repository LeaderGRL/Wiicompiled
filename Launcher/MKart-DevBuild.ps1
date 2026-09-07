[CmdletBinding()]
param(
    [string]$InstallRoot = "C:\Users\jorda\Downloads\MKartPOC\WiiCompiled",
    [int]$Parallel = 20,
    [switch]$BootstrapSourceBuild,
    [switch]$Run,
    [switch]$Diagnostic,
    [switch]$IgnoreDepth
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0

$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$root = [IO.Path]::GetFullPath($InstallRoot)
$workspace = Join-Path $root 'BuildWorkspace'
$toolkit = Join-Path $root 'Toolkit'
$cmake = Join-Path $toolkit 'CMake\bin\cmake.exe'
$build = Join-Path $workspace 'native-build'
$product = Join-Path $root 'Base'

function Assert-Path([string]$Path, [string]$Description, [switch]$Container) {
    $type = if ($Container) { 'Container' } else { 'Leaf' }
    if (-not (Test-Path -LiteralPath $Path -PathType $type)) {
        throw "$Description not found: $Path"
    }
}

function Copy-DevFile([string]$RelativePath) {
    $source = Join-Path $repo $RelativePath
    $destination = Join-Path $workspace $RelativePath
    Assert-Path $source "Repository source"
    [IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
    Write-Host "MKARTDEV: synced $RelativePath"
}

Assert-Path $workspace 'Installed BuildWorkspace' -Container
Assert-Path $toolkit 'Installed Toolkit' -Container
Assert-Path $cmake 'Bundled CMake'
Assert-Path (Join-Path $workspace 'LocalBuild.ps1') 'Installed LocalBuild.ps1'

# Keep this list deliberately narrow. Adding a renderer source here makes it part of the local
# iteration loop without touching translated Nintendo code or user-owned game assets.
$devFiles = @(
    'aurora-main\lib\gfx\modern\scene_renderer.cpp',
    'aurora-main\lib\gfx\modern\scene_renderer.hpp',
    'aurora-main\lib\gx\pipeline.cpp',
    'aurora-main\lib\gx\pipeline.hpp',
    'aurora-main\lib\gfx\CMakeLists.txt'
)
foreach ($relative in $devFiles) {
    $source = Join-Path $repo $relative
    if (Test-Path -LiteralPath $source -PathType Leaf) {
        Copy-DevFile $relative
    }
}

$cache = Join-Path $build 'CMakeCache.txt'
$usesNativePrebuilt = $false
if (Test-Path -LiteralPath $cache -PathType Leaf) {
    $prebuiltLine = Select-String -LiteralPath $cache -Pattern '^MKW_NATIVE_PREBUILT_DIR:PATH=(.*)$' |
        Select-Object -First 1
    if ($null -ne $prebuiltLine) {
        $prebuiltPath = $prebuiltLine.Matches[0].Groups[1].Value
        $usesNativePrebuilt = -not [string]::IsNullOrWhiteSpace($prebuiltPath)
    }
}

if ($BootstrapSourceBuild -or $usesNativePrebuilt -or -not (Test-Path -LiteralPath $cache -PathType Leaf)) {
    Write-Host 'MKARTDEV: bootstrapping an Aurora source build (one-time expensive step)...'
    $toolkitStatePath = Join-Path $root 'toolkit-state.json'
    Assert-Path $toolkitStatePath 'toolkit-state.json'
    $state = Get-Content -LiteralPath $toolkitStatePath -Raw | ConvertFrom-Json

    & (Join-Path $workspace 'LocalBuild.ps1') `
        -Workspace $workspace `
        -Toolkit $toolkit `
        -Profile base `
        -OutputDirectory $product `
        -TranslationFingerprint ([string]$state.TranslationFingerprint) `
        -NativeToolchainFingerprint ([string]$state.NativeToolchainFingerprint) `
        -Parallel $Parallel
    if ($LASTEXITCODE -ne 0) {
        throw "LocalBuild.ps1 failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host 'MKARTDEV: Aurora source build already configured; using incremental Ninja build.'
    & $cmake --build $build --target WiiCompiled --parallel $Parallel
    if ($LASTEXITCODE -ne 0) {
        throw "Incremental native build failed with exit code $LASTEXITCODE"
    }

    $builtExe = Join-Path $build 'WiiCompiled.exe'
    Assert-Path $builtExe 'Built WiiCompiled.exe'
    Copy-Item -LiteralPath $builtExe -Destination (Join-Path $product 'WiiCompiled.exe') -Force
}

Write-Host 'MKARTDEV: build published to Base\WiiCompiled.exe'

if ($Run) {
    $env:AURORA_MODERN_SCENE_POC = '1'
    if ($Diagnostic) {
        $env:AURORA_MODERN_SCENE_DIAGNOSTIC = '1'
    } else {
        Remove-Item Env:AURORA_MODERN_SCENE_DIAGNOSTIC -ErrorAction SilentlyContinue
    }
    if ($IgnoreDepth) {
        $env:AURORA_MODERN_SCENE_IGNORE_DEPTH = '1'
    } else {
        Remove-Item Env:AURORA_MODERN_SCENE_IGNORE_DEPTH -ErrorAction SilentlyContinue
    }
    Remove-Item Env:AURORA_MODERN_RENDERER_POC -ErrorAction SilentlyContinue

    Push-Location $product
    try {
        & (Join-Path $product 'WiiCompiled.exe')
    } finally {
        Pop-Location
    }
}
