@echo off
setlocal enabledelayedexpansion

where uv >nul 2>nul
if errorlevel 1 (
    echo ❌ uv is not installed. Installing...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
)

for /f "tokens=*" %%i in ('uv --version') do echo ✅ uv detected: %%i

echo 📌 Pinning Python to 3.11
call uv python pin 3.11

if not exist .venv (
    echo 🔧 Creating virtual environment
    call uv venv
)

echo 📦 Installing all dependencies
call uv sync --all-groups

echo 🪝 Installing pre-commit hooks
call uv run pre-commit install

echo.
echo ✅ Setup complete!
echo Activate with: .venv\Scripts\activate
echo Run commands via: uv run <cmd>

echo.
endlocal
