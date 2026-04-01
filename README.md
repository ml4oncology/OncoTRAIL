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

```bash
Rscript -e "install.packages(c('pROC'), repos='https://cloud.r-project.org')"
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

Install the prebuilt wheel for CUDA 12.1 to ensure GPU acceleration is enabled:

```bash
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
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

## Pipeline Scripts

All scripts are located under:
```
scripts/paper/
```

## 1. Data Preparation

| Script | Description |
| --- | --- |
| `prep/anchor_note_to_treatment_train.sh` | Anchor Harris Flex EHR notes (training + held-out test set) to tabular treatment data |
| `prep/anchor_note_to_treatment_inference.sh` | Anchor EPIC EHR notes (held-out test set) to tabular treatment data |
| `prep/generate_note_summary.sh` | Generate LLM summaries of clinical notes |
| `prep/combine_note_summary.sh` | Attach generated note summaries to the anchored dataset |

---

## 2. Model Development (Harris Flex EHR)

### Tabular + NLP Models

| Script | Description |
| --- | --- |
| `tabular_nlp/main_tabular_train.sh` | Train tabular and NLP models and evaluate on held-out test set |
| `tabular_nlp/aggregate_results.sh EPR` | Aggregate results across configurations and perform model selection |

---

### Finetuned Language Models

| Script | Description |
| --- | --- |
| `finetuning/main_finetuning_train_test.sh` | Train and evaluate finetuned LLM classifiers |
| `finetuning/post_proc_results.sh EPR` | Aggregate results and identify optimal finetuning configuration |

---

### Prompting-Based LLM Models

#### Prompt Generation and Preprocessing

| Script | Description |
| --- | --- |
| `prompting/generate_prompts.sh` | Generate prompts from clinical notes |
| `prompting/prepare_data.sh` | Split notes into chunks for parallel prompting |

#### Hyperparameter Optimization

| Script | Description |
| --- | --- |
| `prompting/prompting_train_stage1.sh` | Stage 1 hyperparameter search for prompting configurations |
| `prompting/prompting_train_stage2.sh` | Stage 2 hyperparameter refinement |
| `prompting/prompting_train_stage3.sh` | Stage 3 hyperparameter refinement |

> Each stage produces outputs that must be processed separately in the aggregation step below.

---

#### Training Result Aggregation

For each stage (`stage1`, `stage2`, `stage3`), run the following scripts with the corresponding stage argument.

| Script | Description |
| --- | --- |
| `prompting/combine_prompt_results.sh <stage>` | Combine outputs from parallel prompting jobs for a given stage |
| `prompting/compute_stats.sh <stage>` | Compute evaluation metrics for a given stage |
| `prompting/load_aggregate_statistics.sh aggregate <stage>` | Aggregate metrics and identify optimal configuration for a given stage |

#### Held-Out Test Evaluation

| Script | Description |
| --- | --- |
| `prompting/prompting_evaluate.sh EPR_test` | Run prompting evaluation using optimal configuration |
| `prompting/combine_prompt_results.sh EPR_test` | Combine evaluation outputs |
| `prompting/compute_stats.sh EPR_test` | Compute evaluation metrics |
| `prompting/load_aggregate_statistics.sh aggregate EPR_test` | Aggregate evaluation results |

---

## 3. External Evaluation (EPIC EHR)

### Tabular + NLP Models

| Script | Description |
| --- | --- |
| `tabular_nlp/main_inference.sh` | Run inference using optimal configuration on EPIC dataset |
| `tabular_nlp/aggregate_results.sh EPIC` | Aggregate inference results |

### Finetuned Models

| Script | Description |
| --- | --- |
| `finetuning/main_inference.sh` | Run finetuned model inference on EPIC dataset |
| `finetuning/post_proc_results.sh EPIC` | Aggregate inference results |

### Prompting Models

| Script | Description |
| --- | --- |
| `prompting/prompting_evaluate.sh EPIC` | Evaluate prompting model on EPIC dataset |
| `prompting/combine_prompt_results.sh EPIC` | Combine outputs from parallel prompting jobs |
| `prompting/compute_stats.sh EPIC` | Compute evaluation metrics |
| `prompting/load_aggregate_statistics.sh aggregate EPIC` | Aggregate inference results |
| `prompting/load_aggregate_statistics.sh concatenate` | Aggregate results across all data folds |
---

## 4. Postprocessing and Analysis

| Script | Description |
| --- | --- |
| `postproc/aggregate_methods_targets_results.sh` | Aggregate and bootstrap results across all toxicities, methods, and data folds |
| `postproc/generate_word_statistics.sh {EPR/EPIC}` | Compute word frequency and influence statistics |
| `postproc/plot_word_statistics.sh {EPR/EPIC}` | Plot word popularity and alignment between input and model reasoning |
| `postproc/delong_auc_comparison.sh` | Perform DeLong statistical tests for AUC comparison |
| `postproc/plot_physician_characteristics.sh {EPR/EPIC}` | Analyze model sensitivity to physician writing style |
| `postproc/sensitivity_analysis.sh` | Aggregate and bootstrap results across all toxicities, methods, and data folds for healthy patients at baseline |
| `postproc/regress_physician_characteristics.sh {EPR/EPIC}` | Fit Bayesian hierarchical models to quantify sensitivity to physician writing style |
| `postproc/plot_physician_characteristics.sh {EPR/EPIC}` | Plot ICC and regression coefficients for Bayesian models regressed on physician demographics |