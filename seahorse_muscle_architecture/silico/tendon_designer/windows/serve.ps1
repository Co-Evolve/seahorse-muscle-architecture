# Minimal static web server for the Tendon Designer, used when Python is not installed.
# Works with Windows PowerShell 5.1 (built into Windows 10/11) and PowerShell 7.
# Serves the app folder on http://localhost:<port>/ with explicit MIME types
# (the Windows registry often maps .js to text/plain, which breaks ES modules).
param(
    [int]$Port = 8765,
    [string]$Root = (Join-Path $PSScriptRoot "..\app"),
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Root).Path)
if (-not $Root.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
    $Root = $Root + [System.IO.Path]::DirectorySeparatorChar
}

$types = @{
    ".html" = "text/html; charset=utf-8"
    ".js"   = "text/javascript; charset=utf-8"
    ".mjs"  = "text/javascript; charset=utf-8"
    ".wasm" = "application/wasm"
    ".json" = "application/json; charset=utf-8"
    ".css"  = "text/css; charset=utf-8"
    ".svg"  = "image/svg+xml"
    ".png"  = "image/png"
    ".ico"  = "image/x-icon"
    ".stl"  = "application/octet-stream"
    ".xml"  = "application/xml"
    ".md"   = "text/plain; charset=utf-8"
    ".txt"  = "text/plain; charset=utf-8"
    ".csv"  = "text/csv"
}

# Take the first free port starting at $Port.
$listener = $null
for ($p = $Port; $p -lt ($Port + 20); $p++) {
    $candidate = New-Object System.Net.HttpListener
    $candidate.Prefixes.Add("http://localhost:$p/")
    try {
        $candidate.Start()
        $listener = $candidate
        $Port = $p
        break
    } catch {
        $candidate.Close()
    }
}
if ($null -eq $listener) {
    Write-Host "Could not start the web server (ports $Port to $($Port + 19) are busy)."
    exit 1
}

$url = "http://localhost:$Port/"
Write-Host "  Tendon Designer running at $url"
Write-Host "  Keep this window open while you work. Close it to stop the app."
if (-not $NoBrowser) { Start-Process $url }

function Send-Bytes($response, [byte[]]$bytes, [string]$contentType, [bool]$withBody) {
    $response.ContentType = $contentType
    $response.ContentLength64 = $bytes.Length
    if ($withBody) { $response.OutputStream.Write($bytes, 0, $bytes.Length) }
}

try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response
        try {
            $response.Headers.Add("Cache-Control", "no-store, max-age=0")
            $response.Headers.Add("X-Content-Type-Options", "nosniff")
            $withBody = $request.HttpMethod -ne "HEAD"

            $relative = [System.Uri]::UnescapeDataString($request.Url.AbsolutePath).TrimStart("/")
            $relative = $relative.Replace("/", [string][System.IO.Path]::DirectorySeparatorChar)
            $path = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($Root, $relative))
            if ([System.IO.Directory]::Exists($path)) {
                $path = [System.IO.Path]::Combine($path, "index.html")
            }

            $inside = $path.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)
            if ($inside -and [System.IO.File]::Exists($path)) {
                $extension = [System.IO.Path]::GetExtension($path).ToLowerInvariant()
                $contentType = $types[$extension]
                if ($null -eq $contentType) { $contentType = "application/octet-stream" }
                Send-Bytes $response ([System.IO.File]::ReadAllBytes($path)) $contentType $withBody
            } else {
                $response.StatusCode = 404
                Send-Bytes $response ([System.Text.Encoding]::UTF8.GetBytes("Not found")) "text/plain; charset=utf-8" $withBody
            }
        } catch {
            try { $response.StatusCode = 500 } catch {}
        } finally {
            try { $response.Close() } catch {}
        }
    }
} finally {
    $listener.Stop()
    $listener.Close()
}
