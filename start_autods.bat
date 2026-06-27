@echo off
rem AutoDS Agent startup script
rem Kills stale Streamlit processes on ports 8501 and 8502, then launches the app on port 8501.

necho Stopping stale Streamlit processes on ports 8501 and 8502...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R "0.0.0.0:8501 127.0.0.1:8501 \[::\]:8501 0.0.0.0:8502 127.0.0.1:8502 \[::\]:8502"') do (
    for /f "delims= " %%i in ("%%p") do (
        echo Killing PID %%i...
        taskkill /PID %%i /F >nul 2>&1
    )
)
echo Launching AutoDS Agent on http://localhost:8501...
streamlit run app.py --server.port 8501

echo AutoDS Agent started at http://localhost:8501/
pause
