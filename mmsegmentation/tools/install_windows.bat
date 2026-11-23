@echo off
setlocal

if /I not "%CONDA_DEFAULT_ENV%"=="massformer" (
    echo [ERROR] Run conda activate massformer in Anaconda Prompt first.
    exit /b 1
)

pushd "%~dp0.." || exit /b 1
if not exist "requirements\windows-solar.txt" (
    echo [ERROR] Missing requirements\windows-solar.txt in the project directory.
    goto failed
)
if not exist "setup.py" (
    echo [ERROR] Missing setup.py in the project directory.
    goto failed
)

python -c "import sys; from pathlib import Path; sys.exit(0 if Path(sys.prefix).name.lower() == 'massformer' and sys.version_info[:2] == (3, 11) else 1)"
if errorlevel 1 (
    echo [ERROR] The active Python must belong to massformer and be Python 3.11.
    goto failed
)

echo [1/6] Installing build tools...
python -m pip install "setuptools==69.5.1" wheel
if errorlevel 1 goto failed

echo [2/6] Installing NumPy...
python -m pip install "numpy==1.26.4"
if errorlevel 1 goto failed

echo [3/6] Installing CUDA 11.8 PyTorch and TorchVision...
python -m pip install "torch==2.0.1" "torchvision==0.15.2" --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto failed

echo [4/6] Installing the compiled MMCV wheel...
python -m pip install --only-binary=mmcv "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
if errorlevel 1 goto failed

echo [5/6] Installing project requirements...
python -m pip install -r requirements/windows-solar.txt
if errorlevel 1 goto failed

echo [6/6] Installing this repository in editable mode...
python -m pip install -e . --no-deps --no-build-isolation
if errorlevel 1 goto failed

echo [OK] Installation completed in massformer.
popd
exit /b 0

:failed
echo [ERROR] Installation stopped. Resolve the error above before retrying.
popd
exit /b 1
