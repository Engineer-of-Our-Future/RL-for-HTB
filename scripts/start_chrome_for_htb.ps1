# Launch Chrome with a DevTools debugging port and a dedicated HTB profile.
#
# This is the FIRST step of the HTB Academy login flow. Run this in
# PowerShell, log into HTB normally in the Chrome window that appears
# (Google OAuth works fine - it's a real user-driven browser session;
# Cloudflare won't trigger because there are no Playwright automation
# markers), then in a SECOND PowerShell run scripts\htb_academy_login_check.py
# with HTBRL_ACADEMY_CDP=1.
#
# The Chrome profile lives under .local/htb-chrome-profile/ which is
# gitignored. Cookies + tokens persist across launches as long as you keep
# using that profile dir.

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProfileDir  = Join-Path $ProjectRoot ".local\htb-chrome-profile"
$null = New-Item -ItemType Directory -Force -Path $ProfileDir

$ChromeExe = $env:HTBRL_CHROME_PATH
if (-not $ChromeExe) {
    $candidates = @(
        "C:\Program Files\Google\Chrome\Application\chrome.exe",
        "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    )
    $ChromeExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $ChromeExe) {
    Write-Error "No Chrome / Edge install found. Set HTBRL_CHROME_PATH or install Chrome."
    exit 2
}

$Port = 9222
Write-Host "Launching $ChromeExe"
Write-Host "  --remote-debugging-port=$Port"
Write-Host "  --user-data-dir=$ProfileDir"
Write-Host ""
Write-Host "Next: log into https://academy.hackthebox.com/app/dashboard normally."
Write-Host "After you reach the dashboard, run (in another PowerShell):"
Write-Host "    `$env:HTBRL_ACADEMY_CDP = 'http://127.0.0.1:$Port'"
Write-Host "    .venv\Scripts\python.exe scripts\htb_academy_login_check.py"
Write-Host ""

& "$ChromeExe" --remote-debugging-port=$Port --user-data-dir="$ProfileDir" "https://academy.hackthebox.com/app/dashboard"
