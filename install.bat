@echo off
REM ─────────────────────────────────────────────────────────────
REM  ORACLE - Installer (Windows)
REM  Online Readout and Acquisition for Collinear Laser spectroscopy Experiments
REM
REM  Usage:
REM    install.bat            Interactive menu
REM    install.bat full       Full install (uv + dependencies + shortcut)
REM    install.bat rack       Full install plus the instrument libraries
REM    install.bat uv         Install uv only
REM    install.bat shortcut   Create desktop shortcut only
REM ─────────────────────────────────────────────────────────────
setlocal EnableDelayedExpansion

set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

echo ===========================================================
echo   ORACLE - Online Readout ^& Acquisition for CLS Experiments
echo ===========================================================
echo.

set "MODE=%~1"
set "DO_UV=0"
set "DO_SYNC=0"
set "DO_SC=0"
set "SYNC_ARGS="

if /i "%MODE%"=="full"     (set "DO_UV=1" & set "DO_SYNC=1" & set "DO_SC=1" & goto :run)
if /i "%MODE%"=="rack"     (set "DO_UV=1" & set "DO_SYNC=1" & set "DO_SC=1" & set "SYNC_ARGS=--group rack" & goto :run)
if /i "%MODE%"=="uv"       (set "DO_UV=1" & goto :run)
if /i "%MODE%"=="shortcut" (set "DO_SC=1" & goto :run)
if not "%MODE%"=="" (
    echo Unknown option "%MODE%". Use: full ^| rack ^| uv ^| shortcut
    exit /b 1
)

echo Choose an action:
echo   [1] Full install ^(uv + dependencies + desktop shortcut^)
echo   [2] Full install for the beamline PC ^(adds the instrument libraries^)
echo   [3] Install uv only
echo   [4] Create desktop shortcut only ^(assumes .venv exists^)
echo   [q] Quit
echo.
set /p "CHOICE=Selection [1/2/3/4/q]: "
if "!CHOICE!"=="1" (set "DO_UV=1" & set "DO_SYNC=1" & set "DO_SC=1" & goto :run)
if "!CHOICE!"=="2" (set "DO_UV=1" & set "DO_SYNC=1" & set "DO_SC=1" & set "SYNC_ARGS=--group rack" & goto :run)
if "!CHOICE!"=="3" (set "DO_UV=1" & goto :run)
if "!CHOICE!"=="4" (set "DO_SC=1" & goto :run)
if /i "!CHOICE!"=="q" exit /b 0
echo Invalid choice.
exit /b 1

:run
if "!DO_UV!"=="1"   call :install_uv      || goto :fail
if "!DO_SYNC!"=="1" call :sync_deps       || goto :fail
if "!DO_SC!"=="1"   call :create_shortcut || goto :fail

echo.
echo ===========================================================
echo   Done. Launch from terminal:  uv run python gui.py
echo ===========================================================
pause
exit /b 0

:fail
pause
exit /b 1

REM ── Subroutine: install uv if missing ────────────────────────
:install_uv
where uv >nul 2>&1
if !ERRORLEVEL! equ 0 (
    for /f "delims=" %%V in ('uv --version 2^>nul') do set "UV_VER=%%V"
    echo uv already installed: !UV_VER!
    exit /b 0
)
echo uv was not found on PATH.
echo.
set /p "INSTALL_UV=Install uv now via the official Astral installer? [Y/n]: "
if /i "!INSTALL_UV!"=="n" (
    echo.
    echo Cannot continue without uv. Install manually from
    echo   https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)
echo.
echo Installing uv...
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if !ERRORLEVEL! neq 0 (
    echo.
    echo ERROR: uv installation failed.
    exit /b 1
)
REM uv installs to %USERPROFILE%\.local\bin; pick it up for this shell
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo.
    echo uv was installed but is not yet visible in this shell's PATH.
    echo Open a NEW terminal and re-run install.bat.
    exit /b 1
)
exit /b 0

REM ── Subroutine: uv sync (Python + dependencies into .venv) ───
:sync_deps
echo.
echo -----------------------------------------------------------
echo   Installing Python + dependencies via uv sync...
echo -----------------------------------------------------------
echo.
uv sync --managed-python !SYNC_ARGS!
if !ERRORLEVEL! neq 0 (
    echo.
    echo ERROR: uv sync failed. See output above.
    exit /b 1
)
echo.
echo Dependencies installed into .venv
exit /b 0

REM ── Subroutine: create desktop shortcut ──────────────────────
:create_shortcut
set "GUI_PY=%APP_DIR%gui.py"
set "ICO=%APP_DIR%icons\oracle.ico"
set "VENV=%APP_DIR%.venv"
set "PYTHON=!VENV!\Scripts\python.exe"
set "PYTHONW=!VENV!\Scripts\pythonw.exe"

if not exist "!PYTHON!" (
    echo.
    echo ERROR: Project venv not found at .venv\
    echo Run "install.bat full" first to set up uv and dependencies.
    exit /b 1
)

if exist "!PYTHONW!" (
    set "LAUNCH=!PYTHONW!"
    echo Using pythonw.exe ^(no console window^)
) else (
    set "LAUNCH=!PYTHON!"
    echo Note: pythonw.exe not found, shortcut will show a console window
)

REM The Desktop as Windows has it: with OneDrive's folder backup on, it is
REM %USERPROFILE%\OneDrive\Desktop, and %USERPROFILE%\Desktop is not the one shown
set "TARGET="
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "TARGET=%%D"
if not defined TARGET set "TARGET=%USERPROFILE%\Desktop"
set "LNK=!TARGET!\ORACLE.lnk"

REM A .venv under OneDrive syncs, and Files On-Demand can leave its DLLs in
REM the cloud, so Qt or numpy would fail to load: keep it on this device,
REM as install.sh keeps it out of iCloud on macOS
set "IN_ONEDRIVE=0"
for %%O in ("%OneDrive%" "%OneDriveCommercial%" "%OneDriveConsumer%") do (
    if not "%%~O"=="" if /i not "!APP_DIR:%%~O=!"=="!APP_DIR!" set "IN_ONEDRIVE=1"
)
if "!IN_ONEDRIVE!"=="1" (
    attrib +P -U "!VENV!" >nul 2>&1
    attrib +P -U "!VENV!\*" /S /D >nul 2>&1
    echo .venv is under OneDrive: kept on this device ^(always available^)
)
echo Venv:    !VENV!
echo Target:  !LNK!
echo.

powershell -Command ^
    "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('!LNK!'); $s.TargetPath = '!LAUNCH!'; $s.Arguments = '\"!GUI_PY!\"'; $s.WorkingDirectory = '%APP_DIR%'; $s.Description = 'ORACLE - Online Readout and Acquisition for Collinear Laser spectroscopy Experiments'; $s.IconLocation = '!ICO!'; $s.WindowStyle = 1; $s.Save()"
if !ERRORLEVEL! neq 0 (
    echo ERROR: Failed to create shortcut.
    exit /b 1
)
echo Shortcut created: !LNK!
exit /b 0
