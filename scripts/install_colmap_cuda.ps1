$ErrorActionPreference = "Stop"

$release = "4.2.1"
$assetName = "colmap-x64-windows-cuda.zip"
$expectedBytes = 414666626L
$parallelParts = 24
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$toolsRoot = Join-Path $projectRoot "tools"
$downloadRoot = Join-Path $toolsRoot "colmap-download"
$archivePath = Join-Path $toolsRoot $assetName
$installPath = Join-Path $toolsRoot "colmap"
$url = "https://github.com/colmap/colmap/releases/download/$release/$assetName"

if (-not $toolsRoot.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use a tools directory outside the project root."
}

New-Item -ItemType Directory -Force -Path $toolsRoot, $downloadRoot | Out-Null

$chunkSize = [Math]::Ceiling($expectedBytes / $parallelParts)
$downloads = @()
$parts = @()

for ($index = 0; $index -lt $parallelParts; $index++) {
    $start = [int64]($index * $chunkSize)
    $end = [Math]::Min([int64](($index + 1) * $chunkSize - 1), $expectedBytes - 1)
    if ($start -gt $end) { break }

    $partPath = Join-Path $downloadRoot ("part-{0:D3}.bin" -f $index)
    $errorPath = "$partPath.stderr"
    $part = [pscustomobject]@{ Path = $partPath; Start = $start; End = $end; ErrorPath = $errorPath }
    $parts += $part
    $expectedPartBytes = $end - $start + 1
    if ((Test-Path -LiteralPath $partPath) -and (Get-Item -LiteralPath $partPath).Length -eq $expectedPartBytes) {
        continue
    }
    Remove-Item -LiteralPath $partPath -Force -ErrorAction SilentlyContinue

    $arguments = @(
        "-L", "--fail", "--retry", "5", "--retry-delay", "2",
        "--range", "$start-$end", "--output", $partPath, $url
    )
    $process = Start-Process -FilePath "curl.exe" -ArgumentList $arguments -PassThru `
        -WindowStyle Hidden -RedirectStandardError $errorPath
    $downloads += [pscustomobject]@{ Process = $process; Part = $part }
}

while ($downloads.Where({ -not $_.Process.HasExited }).Count -gt 0) {
    $downloaded = (Get-ChildItem -LiteralPath $downloadRoot -Filter "part-*.bin" -File |
        Measure-Object -Property Length -Sum).Sum
    $running = $downloads.Where({ -not $_.Process.HasExited }).Count
    Write-Output ("Downloading COLMAP CUDA: {0:N1}/{1:N1} MB, {2} parts active" -f `
        ($downloaded / 1MB), ($expectedBytes / 1MB), $running)
    Start-Sleep -Seconds 15
    foreach ($download in $downloads) { $download.Process.Refresh() }
}

foreach ($download in $downloads) {
    $process = $download.Process
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        $detail = Get-Content -LiteralPath $download.Part.ErrorPath -Tail 20 -ErrorAction SilentlyContinue
        throw "A COLMAP download part failed with exit code $($process.ExitCode): $detail"
    }
}

foreach ($part in $parts) {
    $expectedPartBytes = $part.End - $part.Start + 1
    $actualPartBytes = (Get-Item -LiteralPath $part.Path).Length
    if ($actualPartBytes -ne $expectedPartBytes) {
        throw "Invalid part size for $($part.Path): expected $expectedPartBytes, got $actualPartBytes."
    }
}

$output = [System.IO.File]::Create($archivePath)
try {
    foreach ($part in $parts) {
        $input = [System.IO.File]::OpenRead($part.Path)
        try { $input.CopyTo($output) } finally { $input.Dispose() }
    }
} finally {
    $output.Dispose()
}

$actualBytes = (Get-Item -LiteralPath $archivePath).Length
if ($actualBytes -ne $expectedBytes) {
    throw "Invalid final archive size: expected $expectedBytes, got $actualBytes."
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    if ($zip.Entries.Count -eq 0) { throw "The COLMAP archive is empty." }
} finally {
    $zip.Dispose()
}

if (Test-Path -LiteralPath $installPath) {
    throw "Install destination already exists: $installPath"
}
Expand-Archive -LiteralPath $archivePath -DestinationPath $installPath

$launcher = Get-ChildItem -LiteralPath $installPath -Recurse -File -Filter "COLMAP.bat" |
    Select-Object -First 1
if (-not $launcher) { throw "COLMAP.bat was not found after extraction." }

Remove-Item -LiteralPath $downloadRoot -Recurse -Force
Remove-Item -LiteralPath $archivePath -Force

Write-Output "COLMAP CUDA installed at $($launcher.FullName)"
