# HLLM-Creator Server Setup Runbook

This runbook is for the GPU server. Follow it step by step. Do not use the
Amazon Electronics data in this phase. The only goal of Phase 1 is to prepare
and verify the Python/CUDA environment for HLLM-Creator.

## What Not To Do Yet

Do not download the 75GB HLLM-Creator checkpoint yet.

Do not download Amazon Electronics data yet.

Do not run full inference yet.

Do not edit model code yet unless an import/runtime check below proves a real
environment issue.

## Expected Machine

The expected server is:

- Linux x86_64
- NVIDIA GPU
- CUDA-capable PyTorch
- bf16-capable GPU preferred
- at least 200GB free disk before downloading models

A100 80GB is the safest GPU target. The released HLLM-Creator setup loads a
large creative LLM plus item/user LLM modules, so small GPUs may fail later even
if the environment installation succeeds.

## Step 0: Enter The Repository

```bash
cd /path/to/HLLM
pwd
ls
```

Expected:

```text
README.md
HLLM_CREATOR_README.md
requirements.txt
code/
reproduce/
```

If these files are not present, stop and pull the correct repository first.

## Step 1: Record Server Hardware

Run:

```bash
nvidia-smi
df -h .
free -h
uname -a
```

Save the output in the server log. The important things to check are:

- GPU name
- driver version
- CUDA version shown by `nvidia-smi`
- free disk space
- CPU RAM

If `nvidia-smi` is not found or shows no GPU, stop. This environment cannot run
HLLM-Creator inference.

## Step 2: Create A Clean Python Environment

Use Python 3.11. This is intentional because the README recommends a FAISS GPU
wheel built for `cp311`.

With conda:

```bash
conda create -n hllm_creator python=3.11 -y
conda activate hllm_creator
python --version
```

Expected:

```text
Python 3.11.x
```

If conda is unavailable, use another isolated Python 3.11 environment. Do not
install these packages into a shared base environment.

## Step 3: Upgrade Build Tools

Run:

```bash
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

Then check:

```bash
python -m pip --version
python - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY
```

## Step 4: Install PyTorch First

The repository README says PyTorch 2.3.1, but `requirements.txt` does not pin
`torch`. Install PyTorch before installing the repo requirements.

Choose exactly one command below.

For CUDA 12.1:

```bash
python -m pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:

```bash
python -m pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

After install, run:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count", torch.cuda.device_count())
    print("gpu0", torch.cuda.get_device_name(0))
    print("bf16_supported", torch.cuda.is_bf16_supported())
PY
```

Expected:

```text
cuda_available True
bf16_supported True
```

If `cuda_available` is `False`, stop and fix PyTorch/CUDA before continuing.

If `bf16_supported` is `False`, record it. Later HLLM-Creator may need config
changes or a different GPU.

## Step 5: Install Repository Requirements

Run from the repository root:

```bash
python -m pip install -r requirements.txt
```

If this fails while building `flash_attn`, first make sure PyTorch imports and
CUDA is visible, then run:

```bash
python -m pip install flash-attn==2.5.9.post1 --no-build-isolation
python -m pip install -r requirements.txt
```

If `fbgemm_gpu==0.5.0` fails, record the full error. HLLM-Creator inference
does not appear to directly require HSTU/fbgemm, so it may be acceptable to
continue after installing the rest manually, but do not silently ignore it.

## Step 6: Install Missing Practical Dependencies

The repo uses parquet files through pandas but does not list `pyarrow`.
Hugging Face download utilities are also needed later.

Run:

```bash
python -m pip install pyarrow "huggingface_hub[cli]"
```

If the server uses Git LFS for model downloads, install it at the system level:

```bash
git lfs version
```

If that command fails, install Git LFS using the server's package manager or ask
the server admin.

## Step 7: Optional FAISS Install

Do not install FAISS unless we need to run clustering ourselves.

For the first official inference smoke test, we should try to use the released
`cluster_256.pt` file from the pretrained model directory. If that works, FAISS
is not needed immediately.

If clustering is needed later and Python is 3.11, use:

```bash
python -m pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Then check:

```bash
python - <<'PY'
import faiss
print("faiss", faiss.__version__)
print("num_gpus", faiss.get_num_gpus())
PY
```

Expected `num_gpus` should be at least 1.

## Step 8: Full Import Check

Run this from the repository root:

```bash
python - <<'PY'
import torch
import transformers
import deepspeed
import lightning
import pandas
import pyarrow
import yaml
import tqdm
import accelerate
import sentencepiece

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("bf16", torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__)
print("deepspeed", deepspeed.__version__)
print("lightning", lightning.__version__)
print("pandas", pandas.__version__)
print("pyarrow", pyarrow.__version__)
print("accelerate", accelerate.__version__)
PY
```

Expected:

- no import errors
- `cuda_available True`
- package versions printed

If this fails, stop and fix the import error before moving to Phase 2.

## Step 9: Repository Code Import Check

Run:

```bash
cd code
python - <<'PY'
from REC.config import Config
from REC.model.HLLM.hllm_creator import HLLM_Creator
from REC.data.dataset.trainset import CreatorProcessor
from REC.data.dataset.collate_fn import customize_rmpad_collate
print("HLLM-Creator code imports OK")
PY
cd ..
```

Expected:

```text
HLLM-Creator code imports OK
```

If this fails, save the full traceback.

## Step 10: Record Installed Package Versions

Run:

```bash
python -m pip freeze > server_pip_freeze_hllm_creator.txt
```

Keep this file on the server. Do not commit it unless explicitly requested.

## Important Notes For Phase 2

The released HLLM-Creator checkpoint is not enough by itself. The code builds
three LLM modules first, then loads the HLLM-Creator checkpoint:

- `item_pretrain_dir`
- `user_pretrain_dir`
- `creative_pretrain_dir`

The README mentions TinyLlama and Qwen3. In Phase 2 we must download or identify
the exact local directories for these base models before official inference can
work.

The official script `reproduce/HLLM_Creator/HLLM_Creator_eval.sh` is a template,
not a ready-to-run command:

- It leaves `item_pretrain_dir`, `user_pretrain_dir`, `creative_pretrain_dir`,
  `train_data_path`, and `test_data_path` empty.
- `generate_usertitle.py` requires `--creative_tokenizer_path`, but the shell
  script does not pass it.
- The shell script passes `--creative_model_path` and `--num_gpus` to
  `generate_usertitle.py`, but that Python script ignores unknown args through
  `parse_known_args`.
- For full data, `generate_usertitle.py` stores per-batch prompt embeddings in a
  list before generation. Use tiny `--limit` values first.

## Phase 1 Success Criteria

Phase 1 is complete only when all are true:

1. `nvidia-smi` works.
2. PyTorch imports and reports `cuda_available True`.
3. `torch.cuda.is_bf16_supported()` is recorded.
4. `transformers`, `deepspeed`, `lightning`, `pandas`, `pyarrow`, `accelerate`,
   and `sentencepiece` import successfully.
5. The repository code import check prints `HLLM-Creator code imports OK`.

After Phase 1 succeeds, proceed to Phase 2: download official weights and run a
small official-data inference smoke test.
