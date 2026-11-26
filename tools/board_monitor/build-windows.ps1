param(
    [string]$Python = 'python',
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '../../artifacts/board_monitor')
)

$ErrorActionPreference = 'Stop'
$arch = & $Python -B -c "import platform, struct; print(platform.system(), struct.calcsize('P') * 8, platform.machine())"
if ($LASTEXITCODE -ne 0 -or $arch -ne 'Windows 64 AMD64') {
    throw 'Use a Windows x64 Python installation.'
}
$packager = & $Python -B -c "from importlib.metadata import version; print(version('pyinstaller'))"
if ($LASTEXITCODE -ne 0 -or $packager -ne '6.22.3') {
    throw 'Install PyInstaller 6.22.3 in the build environment.'
}
$source = Join-Path $PSScriptRoot 'monitor.py'
$version = & $Python -B $source --version
if ($LASTEXITCODE -ne 0 -or $version -notmatch '^\d+\.\d+\.\d+$') {
    throw 'Could not read the x.y.z application version.'
}
$basePrefix = & $Python -B -c 'import sys; print(sys.base_prefix)'
if ($LASTEXITCODE -ne 0) { throw 'Could not locate the Python runtime.' }

$name = "K230-Board-Monitor-$version-win-x64"
$output = [IO.Path]::GetFullPath($OutputDirectory)
$destination = Join-Path $output "$name.exe"
$checksum = Join-Path $output "$name.exe.sha256"
if ((Test-Path -LiteralPath $destination) -or (Test-Path -LiteralPath $checksum)) {
    throw "Build output already exists in $output."
}

$buildRoot = Join-Path ([IO.Path]::GetTempPath()) "srtp-monitor-build-$([guid]::NewGuid().ToString('N'))"
$resources = Join-Path $buildRoot 'licenses'
New-Item -ItemType Directory -Path $resources -Force | Out-Null
$pythonLicense = @('LICENSE_PYTHON.txt', 'LICENSE.txt') |
    ForEach-Object { Join-Path $basePrefix $_ } |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if ($pythonLicense) { Copy-Item -LiteralPath $pythonLicense -Destination (Join-Path $resources 'Python.txt') }
foreach ($runtime in @('tcl', 'tk')) {
    $terms = @("tcl/$($runtime)8.6/license.terms", "Library/lib/$($runtime)8.6/license.terms") |
        ForEach-Object { Join-Path $basePrefix $_ } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
    if (-not $terms -and $runtime -eq 'tcl') {
        # Anaconda can retain the Tcl license only in the installed package cache.
        $meta = Join-Path $basePrefix 'conda-meta'
        if (Test-Path -LiteralPath $meta -PathType Container) {
            $package = Get-ChildItem -LiteralPath $meta -Filter 'tk-*.json' -File | Select-Object -First 1
            if ($package) {
                $cached = Join-Path $basePrefix "pkgs/$($package.BaseName)/info/licenses"
                if (Test-Path -LiteralPath $cached -PathType Container) {
                    $terms = Get-ChildItem -LiteralPath $cached -Filter 'license.terms' -File -Recurse |
                        Where-Object { $_.Directory.Name -like 'tcl*' } |
                        Select-Object -First 1 -ExpandProperty FullName
                }
            }
        }
    }
    if ($terms) { Copy-Item -LiteralPath $terms -Destination (Join-Path $resources "$runtime.txt") }
}
$pyInstallerLicense = & $Python -B -c "from importlib.metadata import distribution; d=distribution('pyinstaller'); print(next(d.locate_file(f) for f in d.files if f.name == 'COPYING.txt'))"
if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $pyInstallerLicense -PathType Leaf)) {
    Copy-Item -LiteralPath $pyInstallerLicense -Destination (Join-Path $resources 'PyInstaller.txt')
}

$peVersion = ($version.Split('.') + '0') -join ', '
$versionFile = Join-Path $buildRoot 'file-version.txt'
[IO.File]::WriteAllText($versionFile, @"
VSVersionInfo(
  ffi=FixedFileInfo(filevers=($peVersion), prodvers=($peVersion), mask=0x3f, flags=0x0,
    OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('FileDescription', 'K230 Board Monitor'),
      StringStruct('FileVersion', '$version'),
      StringStruct('ProductName', 'K230 Board Monitor'),
      StringStruct('ProductVersion', '$version')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ])
"@)

$distDir = Join-Path $buildRoot 'dist'
$arguments = @(
    '-B', '-m', 'PyInstaller', '--noconfirm', '--onefile', '--windowed', '--noupx',
    '--name', $name, '--distpath', $distDir, '--workpath', (Join-Path $buildRoot 'work'),
    '--specpath', $buildRoot, '--version-file', $versionFile
)
if (Get-ChildItem -LiteralPath $resources -File) { $arguments += @('--add-data', "$($resources):licenses") }
& $Python @arguments $source
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed. Build files remain in $buildRoot." }

New-Item -ItemType Directory -Path $output -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $distDir "$name.exe") -Destination $destination
$hash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($checksum, "$hash  $name.exe`n")
Write-Output $destination, $checksum
