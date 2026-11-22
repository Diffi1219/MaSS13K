# Run after: conda create -n massformer python=3.10; conda activate massformer
$ErrorActionPreference = 'Stop'
if ((python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") -ne '3.10') {
    throw 'Activate a Python 3.10 conda environment first.'
}
Push-Location $PSScriptRoot
try {
    python -m pip install 'pip==24.3.1' 'setuptools==75.3.2' wheel
    if ($LASTEXITCODE -ne 0) { throw 'Build tools installation failed.' }
    python -m pip install 'torch==2.1.0' 'torchvision==0.16.0' --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    python -m pip install -r requirements/pv-windows.txt
    if ($LASTEXITCODE -ne 0) { throw 'Project dependencies installation failed.' }
    # Binary only: fail clearly if this platform has no matching CUDA wheel.
    python -m pip install 'mmcv==2.1.0' --only-binary=mmcv -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
    if ($LASTEXITCODE -ne 0) { throw 'MMCV wheel installation failed. Do not replace it with mmcv-lite.' }
    python -m pip install --no-deps --no-build-isolation -e .
    if ($LASTEXITCODE -ne 0) { throw 'Local MMSegmentation installation failed.' }
    python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
    Write-Host 'Installed. Next run: python tools/pv.py check --full-data --smoke'
}
finally {
    Pop-Location
}
