@echo off
REM Mo ung dung M31 tren Windows. Can cai thu vien truoc (xem README.md).
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m streamlit run app.py
pause
