# Compatible with Windows PowerShell 5.1 and PowerShell 7.
& {
function Install-Lana {
    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'
    $version = '@LANA_VERSION@'
    $base = "https://github.com/oponite/lana/releases/download/v$version"
    if ($env:OS -ne 'Windows_NT') { throw 'This installer supports Windows only.' }
    $cpu = [int](@(Get-CimInstance Win32_Processor)[0].Architecture)
    $architecture = switch ($cpu) { 9 { 'x64' } 12 { 'arm64' } default { throw 'Unsupported Windows processor.' } }
    if (-not $env:LOCALAPPDATA) { throw 'LOCALAPPDATA must be set.' }
    $parent = Join-Path $env:LOCALAPPDATA 'Programs'
    $prefix = Join-Path $parent 'Lana'
    $previous = "$prefix.previous"
    $lock = Join-Path $parent '.lana-install.lock'
    [IO.Directory]::CreateDirectory($parent) | Out-Null
    # CreateNew is atomic; an existing lock is never overwritten.
    try { $handle = [IO.File]::Open($lock, 'CreateNew', 'Write', 'None') }
    catch { throw "Installer lock exists or cannot be created: $lock. Inspect it before retrying." }
    $work = Join-Path $parent ('.lana-install-' + [Guid]::NewGuid().ToString('N'))
    $promoted = $false; $moved = $false; $committed = $false; $pathChanged = $false
    $oldUserPath = $null
    $registry = $null; $pathExisted = $false; $pathKind = [Microsoft.Win32.RegistryValueKind]::String
    $oldProcessPath = $env:Path
    $savedEnvironment = @{}
    try {
        $registry = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Environment')
        $pathExisted = @($registry.GetValueNames() | Where-Object { $_ -ieq 'Path' }).Count -ne 0
        if ($pathExisted) {
            $pathKind = $registry.GetValueKind('Path')
            if ($pathKind -notin @([Microsoft.Win32.RegistryValueKind]::String, [Microsoft.Win32.RegistryValueKind]::ExpandString)) {
                throw 'User PATH is not a string; refusing to replace it.'
            }
            $oldUserPath = $registry.GetValue('Path', $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        }
        [IO.Directory]::CreateDirectory($work) | Out-Null
        if (Test-Path -LiteralPath $previous) { throw "Previous installation needs inspection: $previous" }
        if (Test-Path -LiteralPath $prefix) {
            if ((Get-Item -LiteralPath $prefix -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing a linked installation: $prefix"
            }
            $marker = Join-Path $prefix '.lana-install'
            if (-not (Test-Path -LiteralPath $marker -PathType Leaf) -or
                [IO.File]::ReadAllText($marker).Trim() -ne 'lana-installer-v1') {
                throw "Refusing to replace an unrelated directory: $prefix"
            }
        }
        $asset = "lana-$version-windows-$architecture.zip"
        foreach ($name in @($asset, 'SHA256SUMS')) {
            for ($attempt = 0; ; $attempt++) {
                try {
                    Invoke-WebRequest -UseBasicParsing -Uri "$base/$name" -OutFile (Join-Path $work $name) -TimeoutSec 180 | Out-Null
                    break
                } catch { if ($attempt -ge 2) { throw } }
            }
        }
        $lines = @([IO.File]::ReadAllLines((Join-Path $work 'SHA256SUMS')) | Where-Object {
            $_ -cmatch ('(^|\s)' + [regex]::Escape($asset) + '$')
        })
        if ($lines.Count -ne 1 -or $lines[0] -cnotmatch ('^[a-fA-F0-9]{64}  ' + [regex]::Escape($asset) + '$')) {
            throw 'Missing or invalid archive checksum.'
        }
        $archive = Join-Path $work $asset
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ine $lines[0].Substring(0, 64)) {
            throw 'Archive checksum mismatch.'
        }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        if (-not ('Oponite.LanaInstaller.Notification' -as [type])) {
            Add-Type @'
using System;
using System.Runtime.InteropServices;
namespace Oponite.LanaInstaller {
    public static class Notification {
        [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern IntPtr SendMessageTimeout(IntPtr window, uint message, UIntPtr parameter,
            string value, uint flags, uint timeout, out UIntPtr result);
    }
}
'@
        }
        $zip = [IO.Compression.ZipFile]::OpenRead($archive)
        try {
            foreach ($entry in $zip.Entries) {
                $name = $entry.FullName
                if ($name -cnotmatch '^[a-zA-Z0-9_./-]+$' -or $name.StartsWith('/') -or
                    $name -cmatch '(^|/)\.\.(/|$)' -or
                    $name -cnotmatch '^(bin/.*|share/.*|version\.txt)$' -or
                    (($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000) {
                    throw "Unsafe archive entry: $name"
                }
            }
        } finally { $zip.Dispose() }
        $bundle = Join-Path $work 'bundle'
        [IO.Compression.ZipFile]::ExtractToDirectory($archive, $bundle)
        foreach ($name in @('bin/lana.exe', 'bin/lana-compiler.labc', 'share/lana/stdlib/core.lana', 'share/doc/lana/LICENSE', 'version.txt')) {
            if (-not (Test-Path -LiteralPath (Join-Path $bundle $name) -PathType Leaf)) { throw "Missing archive asset: $name" }
        }
        if ([IO.File]::ReadAllText((Join-Path $bundle 'version.txt')).Trim() -ne $version) { throw 'Archive version mismatch.' }
        foreach ($name in @('LANA_COMPILER_LABC', 'LANA_STDLIB_DIR')) {
            $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
            [Environment]::SetEnvironmentVariable($name, $null, 'Process')
        }
        $check = Join-Path $work 'check.lana'
        [IO.File]::WriteAllText($check, 'import "std/core" as core; assert(resolve(core.distribution([[7, 1]])) == 7, "installed stdlib");')
        $executable = Join-Path $bundle 'bin/lana.exe'
        Push-Location $work
        try {
            $report = (& $executable version | Out-String).Trim()
            if ($LASTEXITCODE -ne 0 -or -not $report.StartsWith("Lana $version (LABC v2,")) { throw 'Executable version mismatch.' }
            & $executable run $check
            if ($LASTEXITCODE -ne 0) { throw 'Installed compiler/stdlib check failed.' }
        } finally { Pop-Location }
        [IO.File]::WriteAllText((Join-Path $bundle '.lana-install'), 'lana-installer-v1')
        if (Test-Path -LiteralPath $prefix) {
            foreach ($file in Get-ChildItem -LiteralPath $prefix -Recurse -File -Force) {
                try { $probe = [IO.File]::Open($file.FullName, 'Open', 'Read', 'None'); $probe.Dispose() }
                catch { throw "Close programs using the existing Lana installation, then retry. $($_.Exception.Message)" }
            }
            [IO.Directory]::Move($prefix, $previous); $moved = $true
        }
        [IO.Directory]::Move($bundle, $prefix); $promoted = $true
        $bin = Join-Path $prefix 'bin'
        $entries = @($oldUserPath -split ';' | Where-Object {
            [Environment]::ExpandEnvironmentVariables($_).TrimEnd('\') -ieq $bin.TrimEnd('\')
        })
        if ($entries.Count -eq 0) {
            $pathChanged = $true
            $newUserPath = if ($oldUserPath) { "$bin;$oldUserPath" } else { $bin }
            $registry.SetValue('Path', $newUserPath, $pathKind)
        }
        if (@($env:Path -split ';' | Where-Object { $_.TrimEnd('\') -ieq $bin.TrimEnd('\') }).Count -eq 0) {
            $env:Path = "$bin;$env:Path"
        }
        $committed = $true
        Write-Host "Installed Lana $version at $prefix"
        Write-Host 'User PATH configured. Restart other terminals to pick up the change.'
        $resolved = Get-Command lana -ErrorAction SilentlyContinue
        if ($resolved -and $resolved.Source -ine (Join-Path $bin 'lana.exe')) {
            Write-Warning "Another lana takes precedence: $($resolved.Source)"
        }
    } finally {
        $recovered = $true
        if (-not $committed) {
            try {
                if ($pathChanged) {
                    if ($pathExisted) { $registry.SetValue('Path', $oldUserPath, $pathKind) }
                    else { $registry.DeleteValue('Path', $false) }
                }
            } catch {
                $recovered = $false
                Write-Warning "Could not restore user PATH. Previous value: $oldUserPath. $($_.Exception.Message)"
            }
            $env:Path = $oldProcessPath
            try {
                if ($promoted) { Remove-Item -LiteralPath $prefix -Recurse -Force }
                if ($moved) { [IO.Directory]::Move($previous, $prefix) }
            } catch {
                $recovered = $false
                Write-Warning "Recovery required: retain $work and $previous and restore the old bundle before retrying. $($_.Exception.Message)"
            }
        }
        foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process') }
        if ($registry) { $registry.Dispose() }
        if ($pathChanged) {
            try {
                $notificationResult = [UIntPtr]::Zero
                [Oponite.LanaInstaller.Notification]::SendMessageTimeout([IntPtr]0xffff, 0x1a, [UIntPtr]::Zero,
                    'Environment', 2, 5000, [ref]$notificationResult) | Out-Null
            } catch { Write-Warning "PATH was saved; restart your session if other terminals do not see it. $($_.Exception.Message)" }
        }
        if ($committed -and $moved) {
            try { Remove-Item -LiteralPath $previous -Recurse -Force }
            catch { Write-Warning "Old bundle needs manual removal after inspection: $previous" }
        }
        $handle.Dispose()
        if ($recovered) {
            if (Test-Path -LiteralPath $work) { Remove-Item -LiteralPath $work -Recurse -Force }
            Remove-Item -LiteralPath $lock -Force
        }
    }
}
Install-Lana
}
