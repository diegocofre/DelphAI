@echo off
REM Activar entorno virtual
python -m venv venv
.\venv\Scripts\activate

REM Instalar dependencias
pip install -r requirements.txt

REM Ejecutar el script
python tu_script.py

pause
