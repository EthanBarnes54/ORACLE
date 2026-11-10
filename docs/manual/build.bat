@echo off
REM ─────────────────────────────────────────────────────────────
REM  Build the ORACLE user manual PDF (Windows + MiKTeX).
REM  Output: build\manual.pdf
REM ─────────────────────────────────────────────────────────────
setlocal
cd /d "%~dp0"

where latexmk >nul 2>&1
if %ERRORLEVEL% equ 0 (
    latexmk -pdf -interaction=nonstopmode -outdir=build manual.tex
) else (
    where tectonic >nul 2>&1
    if errorlevel 1 (
        echo ERROR: neither latexmk nor tectonic is on PATH.
        echo Install MiKTeX, or Tectonic, then re-run.
        pause
        exit /b 1
    )
    if not exist build mkdir build
    tectonic --outdir build manual.tex
)
set RC=%ERRORLEVEL%
if %RC% neq 0 (
    echo.
    echo Build FAILED. See build\manual.log for details.
    pause
    exit /b %RC%
)

echo.
echo ===========================================================
echo   PDF built: build\manual.pdf
echo ===========================================================
exit /b 0
