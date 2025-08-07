@echo off
REM Batch script to run Python scripts with TensorFlow environment
REM Usage: run_with_tensorflow.bat script_name.py

if "%1"=="" (
    echo Usage: run_with_tensorflow.bat script_name.py
    echo Example: run_with_tensorflow.bat test_enhanced_system.py
    exit /b 1
)

echo Activating TensorFlow environment...
C:\tf\Scripts\activate.bat
echo Running %1 with TensorFlow...
C:\tf\Scripts\python.exe %1