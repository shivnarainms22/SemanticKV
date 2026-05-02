#!/bin/bash
# runpod_setup.sh — run once after spinning up instance

set -e

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Download Llama 3 8B Instruct (requires HF token)
# Replace YOUR_HF_TOKEN or set HF_TOKEN env var
python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    token=os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN'),
    local_dir='./models/llama3-8b-instruct'
)
"

# Verify GPU
python -c "
import torch
print('GPU:', torch.cuda.get_device_name(0))
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
print('torch:', torch.__version__)
import transformers
print('transformers:', transformers.__version__)
"

echo "Setup complete."
