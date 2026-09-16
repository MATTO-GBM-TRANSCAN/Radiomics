@echo off
setlocal EnableExtensions

REM Runs MATTO-GBM pretrained model inference from a JSON config.

REM --- Edit these four lines ------------------------------------------------
set "INPUT_DIR=C:\path\to\your\input"
set "OUTPUT_DIR=C:\path\to\your\output"
set "MODELS_DIR=C:\path\to\your\models"
set "CONFIG_FILE=inference_config.json"
REM ---------------------------------------------------------------------------

if not exist "%INPUT_DIR%" (
    echo INPUT_DIR does not exist: "%INPUT_DIR%"
    exit /b 1
)

if not exist "%OUTPUT_DIR%" (
    echo OUTPUT_DIR does not exist: "%OUTPUT_DIR%"
    exit /b 1
)

if not exist "%MODELS_DIR%" (
    echo MODELS_DIR does not exist: "%MODELS_DIR%"
    echo Set MODELS_DIR to the folder containing the model ZIP files or extracted model folders.
    exit /b 1
)

docker run --rm ^
  -v "%INPUT_DIR%:/input" ^
  -v "%OUTPUT_DIR%:/output" ^
  -v "%MODELS_DIR%:/models" ^
  matto-radiomics ^
  sh -c "umask 000 && python MATTO-GBM_Models/code/run_inference.py '/input/%CONFIG_FILE%'"

endlocal

