@echo off
echo Starting FastAPI backend...
cd /d "%~dp0backend"
pip install -r requirements.txt --quiet
uvicorn main:app --reload --host 0.0.0.0 --port 8000
