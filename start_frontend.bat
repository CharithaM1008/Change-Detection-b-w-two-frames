@echo off
echo Starting React frontend...
cd /d "%~dp0frontend"
call npm install
call npm run dev
