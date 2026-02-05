Write-Host "Creating Conda environment 'unsloth_env'..."

conda create --name unsloth_env python=3.11 -y

if ($LASTEXITCODE -ne 0) {
    Write-Error "Error creating Conda environment."
    exit 1
}

Write-Host "Run Conda init (if not yet performed)"
conda init

Write-Host "Activating Conda environment 'unsloth_env'..."
conda activate unsloth_env

if ($LASTEXITCODE -ne 0) {
    Write-Error "Error activating Conda environment."
    exit 1
}

# Write-Host "Installing Pytorch via pip..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install Pytorch."
    exit 1
}

Write-Host "Installing required packages via pip..."
pip install --no-deps trl peft accelerate bitsandbytes

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install required packages."
    exit 1
}

Write-Host "Installing xformers..."
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install xformers."
    exit 1
}

Write-Host "Installing Unsloth from Git repository..."
pip install --upgrade --no-cache-dir git+https://github.com/unslothai/unsloth.git

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install Unsloth."
    exit 1
}

Write-Host "Installing Unsloth-zoo"
pip install unsloth-zoo

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install Unsloth-zoo."
    exit 1
}

Write-Host "Installing Triton from wheel file..."
pip install -U triton-windows

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install Triton."
    exit 1
}

Write-Host "Installation completed successfully!"
Write-Host "You can activate the environment with: conda activate unsloth_env"