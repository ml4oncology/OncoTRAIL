# OncoTRAIL: Oncology Toxicity Risk prediction using Artificial Intelligence and Language models

## Installation

### Prerequisites
- Linux x86_64
- CUDA 12.x
- conda or mamba (recommended)

### 1. Create the base environment

```bash
mamba create -n OncoTRAIL python=3.11 -c conda-forge
conda activate OncoTRAIL
```

### 2. Install PyTorch with CUDA support

```bash
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 \
    -c pytorch -c nvidia -c conda-forge
```

### 3. Install CUDA toolkit, R, and rpy2

```bash
mamba install cuda-toolkit r-base rpy2 ipykernel ipywidgets \
    -c conda-forge -c nvidia
```

### 4. Install Python dependencies

```bash
pip install \
    pandas numpy scipy polars-lts-cpu \
    scikit-learn statsmodels \
    xgboost lightgbm \
    "autogluon.tabular" \
    shap bayesian-optimization \
    bambi arviz pymc \
    matplotlib seaborn adjustText \
    transformers accelerate datasets trl \
    bitsandbytes \
    sentencepiece tiktoken nltk \
    openai \
    vllm \
    submitit \
    pynvml psutil tqdm \
    pydantic python-dotenv
```

### 5. Install llama-cpp-python with GPU support

This step requires setting the correct `CMAKE_ARGS` flag before installation to ensure GPU acceleration is enabled. Without this, llama-cpp will fall back to CPU-only mode and inference will be very slow.

```bash
export CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python==0.3.8 --no-cache-dir --force-reinstall
```

To verify GPU is working correctly:
```python
from llama_cpp import Llama
llm = Llama(model_path='path/to/your/model.gguf', n_gpu_layers=-1, verbose=True)
# Look for: "ggml_cuda_init: found X CUDA devices" in the output
# All layers should show "assigned to device CUDA0"
```

### 6. Install OncoTRAIL

```bash
# Navigate to the repo root (where pyproject.toml lives)
cd /path/to/oncotrail

pip install -e .
```

### 7. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name OncoTRAIL --display-name "OncoTRAIL"
```