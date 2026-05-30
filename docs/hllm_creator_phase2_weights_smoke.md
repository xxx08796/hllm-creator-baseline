# HLLM-Creator Phase 2: Download Weights And Run Official Smoke Test

This runbook is for the GPU server after Phase 1 environment validation.

Phase 2 goal: download the released HLLM-Creator checkpoint and run the smallest
official smoke test with official synthetic data. Do not use Amazon Electronics
data in this phase.

## Current Phase 1 Assumptions

Before starting, the server should already have:

- repo at `/home/yuanhanyang.yhy/hllm-creator-baseline`
- conda env `hllm_creator`
- PyTorch CUDA import working
- core imports working: `torch`, `transformers`, `deepspeed`, `lightning`,
  `pandas`, `pyarrow`, `accelerate`, `sentencepiece`, `torch_geometric`

Known acceptable Phase 1 exceptions:

- `tensorflow_cpu` can stay uninstalled. HLLM-Creator inference is PyTorch based.
- `fbgemm_gpu` can stay uninstalled unless running HSTU/IDNet.

Open issue:

- `flash_attn` is not just a speed package in this repo. Some HLLM code has
  fallback paths, but `modeling_mistral.py` imports `flash_attn.bert_padding`
  at module import time. If repo import fails because of this, stop and report
  the exact error. Do not edit code silently.

## What Not To Do Yet

Do not use the user's Amazon Electronics data.

Do not start a full large evaluation.

Do not train or fine-tune anything.

Do not delete model files once download starts unless explicitly asked. The
checkpoint is large.

## Official Files

Official Hugging Face repo:

```text
ByteDance/HLLM
```

HLLM-Creator checkpoint directory:

```text
HLLM_Creator/pretrained_model
```

Official synthetic data directory:

```text
HLLM_Creator/fake_train_data
```

As of 2026-05-30, the HLLM-Creator pretrained directory is about 75.8GB and
contains at least:

```text
zero3_merge_states.pt      # about 41.9GB, full HLLM-Creator checkpoint
pytorch_model.bin          # about 32.8GB, creative LLM hf-format weights
cluster_256.pt
cluster_64.pt
cluster_8.pt
rank0_user_emb.pt ... rank7_user_emb.pt
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
merges.txt
vocab.json
```

The synthetic data directory contains:

```text
train.parquet
```

## Step 0: Enter Repo And Env

```bash
cd /home/yuanhanyang.yhy/hllm-creator-baseline
source /opt/conda/etc/profile.d/conda.sh
conda activate hllm_creator
pwd
python --version
```

Expected:

```text
/home/yuanhanyang.yhy/hllm-creator-baseline
Python 3.11.x
```

## Step 1: Record Disk And Inode Before Download

```bash
df -h .
df -i .
df -h /data/oss_bucket_0/yhy
df -i /data/oss_bucket_0/yhy
du -sh . 2>/dev/null || true
```

Stop if `/data/oss_bucket_0/yhy` free disk is below 120GB. The pretrained
folder is about 75.8GB, and download tools need temporary space.

If inode free count is very low, report it before downloading.

## Step 2: Check Hugging Face Download Tool

```bash
python - <<'PY'
import huggingface_hub
print("huggingface_hub", huggingface_hub.__version__)
PY

huggingface-cli --help | head
```

If `huggingface_hub` or `huggingface-cli` is missing:

```bash
python -m pip install "huggingface_hub[cli]"
```

## Step 3: Download Official HLLM-Creator Weights

Use `hf download` if available:

```bash
mkdir -p /data/oss_bucket_0/yhy/ByteDance_HLLM

hf download ByteDance/HLLM \
  --repo-type model \
  --include 'HLLM_Creator/pretrained_model/*' \
  --local-dir /data/oss_bucket_0/yhy/ByteDance_HLLM \
  --local-dir-use-symlinks False \
  --resume-download
```

If the `hf` command is unavailable, use `huggingface-cli`:

```bash
mkdir -p /data/oss_bucket_0/yhy/ByteDance_HLLM

huggingface-cli download ByteDance/HLLM \
  --repo-type model \
  --include 'HLLM_Creator/pretrained_model/*' \
  --local-dir /data/oss_bucket_0/yhy/ByteDance_HLLM \
  --local-dir-use-symlinks False \
  --resume-download
```

If download is slow or interrupted, rerun the same command. It should resume.

Expected model path after download:

```text
/data/oss_bucket_0/yhy/ByteDance_HLLM/HLLM_Creator/pretrained_model
```

## Step 4: Verify Downloaded Weight Files

```bash
MODEL_DIR=/data/oss_bucket_0/yhy/ByteDance_HLLM/HLLM_Creator/pretrained_model

ls -lh "$MODEL_DIR"
du -sh "$MODEL_DIR"

python - <<'PY'
from pathlib import Path
p = Path("/data/oss_bucket_0/yhy/ByteDance_HLLM/HLLM_Creator/pretrained_model")
required = [
    "zero3_merge_states.pt",
    "pytorch_model.bin",
    "cluster_256.pt",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
]
missing = [x for x in required if not (p / x).exists()]
print("model_dir", p.resolve())
print("missing", missing)
for x in required:
    f = p / x
    if f.exists():
        print(x, f.stat().st_size)
raise SystemExit(1 if missing else 0)
PY
```

Expected:

```text
missing []
```

Important size sanity checks:

- `zero3_merge_states.pt` should be tens of GB.
- `pytorch_model.bin` should be tens of GB.
- `cluster_256.pt` should exist.

If either large file is tiny, the download is incomplete.

## Step 5: Download Official Synthetic Data

```bash
hf download ByteDance/HLLM \
  --repo-type model \
  --include 'HLLM_Creator/fake_train_data/train.parquet' \
  --local-dir /data/oss_bucket_0/yhy/ByteDance_HLLM \
  --local-dir-use-symlinks False \
  --resume-download
```

Fallback:

```bash
huggingface-cli download ByteDance/HLLM \
  --repo-type model \
  --include 'HLLM_Creator/fake_train_data/train.parquet' \
  --local-dir /data/oss_bucket_0/yhy/ByteDance_HLLM \
  --local-dir-use-symlinks False \
  --resume-download
```

Verify:

```bash
FAKE_DATA=/data/oss_bucket_0/yhy/ByteDance_HLLM/HLLM_Creator/fake_train_data/train.parquet

python - <<'PY'
import pandas as pd
p = "/data/oss_bucket_0/yhy/ByteDance_HLLM/HLLM_Creator/fake_train_data/train.parquet"
df = pd.read_parquet(p)
print("rows", len(df))
print("columns", list(df.columns))
print(df.head(2).to_string())
PY
```

Expected columns should include:

```text
user_profile
original_title
original_description
prompt1
prompt2
response
title_list
item_id_list
```

## Step 6: Confirm The Pretrained LLM Directory Choice

The eval script needs these three dirs:

```text
item_pretrain_dir
user_pretrain_dir
creative_pretrain_dir
```

For the official released checkpoint, use the downloaded `pretrained_model`
directory for `creative_pretrain_dir` because it contains the tokenizer/config
and `pytorch_model.bin` extracted for the creative LLM.

For `item_pretrain_dir` and `user_pretrain_dir`, first try the same
`pretrained_model` directory only for a tiny import/model-load smoke test.

If model loading fails due to architecture mismatch, stop and report the exact
error. The next likely action is to download the base LLMs named by the official
config/README, but do not guess silently.

## Step 7: Minimal Repo Import Check With Current Package State

This checks whether missing `flash_attn` blocks HLLM-Creator imports.

```bash
cd /home/yuanhanyang.yhy/hllm-creator-baseline/code

python - <<'PY'
import sys
print("python", sys.executable)
from REC.model.HLLM.hllm_creator import HLLM_Creator
from REC.config import Config
from REC.data.dataset.trainset import CreatorProcessor
print("HLLM_CREATOR_IMPORT_OK")
PY
```

If this fails with:

```text
ModuleNotFoundError: No module named 'flash_attn'
```

then `flash_attn` is currently a blocking import dependency. Stop and report.
Do not continue to model loading until we decide whether to install
`flash_attn` or patch the Mistral import to be optional.

If this passes, continue.

## Step 8: Tiny Model Construction Smoke Test

This test attempts to instantiate HLLM-Creator and load the released checkpoint.
It is intentionally tiny: no generation yet.

```bash
cd /home/yuanhanyang.yhy/hllm-creator-baseline/code

MODEL_DIR=/data/oss_bucket_0/yhy/ByteDance_HLLM/HLLM_Creator/pretrained_model
CKPT=$MODEL_DIR/zero3_merge_states.pt

MODEL_DIR="$MODEL_DIR" CKPT="$CKPT" python - <<'PY'
import os
import torch
from REC.config import Config
from REC.model.HLLM.hllm_creator import HLLM_Creator

model_dir = os.environ["MODEL_DIR"]
ckpt = os.environ["CKPT"]

config = Config(config_file_list=["overall/LLM_deepspeed.yaml", "HLLM_Creator/HLLM_Creator.yaml"])
config.parse_extra_args([
    "--MAX_ITEM_LIST_LENGTH", "50",
    "--MAX_TEXT_LENGTH", "64",
    "--item_pretrain_dir", model_dir,
    "--user_pretrain_dir", model_dir,
    "--creative_pretrain_dir", model_dir,
    "--train_path", "/dev/null",
    "--gradient_checkpointing", "True",
    "--use_ft_flash_attn", "False",
])

print("constructing model")
model = HLLM_Creator(config, dataload=None)
print("loading ckpt header")
state = torch.load(ckpt, map_location="cpu")
print("state_keys", len(state))
msg = model.load_state_dict(state, strict=False)
print("missing_keys", len(msg.missing_keys))
print("unexpected_keys", len(msg.unexpected_keys))
print("MODEL_LOAD_SMOKE_OK")
PY
```

Expected:

```text
MODEL_LOAD_SMOKE_OK
```

If this OOMs on CPU RAM or GPU RAM, report exact memory error.

If this fails because `item_pretrain_dir` or `user_pretrain_dir` is wrong,
report exact missing/config error.

## Step 9: Official Generation Smoke Test On Fake Data

Only run this after Step 8 passes.

Use a very small limit and one GPU:

```bash
cd /home/yuanhanyang.yhy/hllm-creator-baseline/code

MODEL_DIR=/data/oss_bucket_0/yhy/ByteDance_HLLM/HLLM_Creator/pretrained_model
FAKE_DATA=/data/oss_bucket_0/yhy/ByteDance_HLLM/HLLM_Creator/fake_train_data/train.parquet
OUT_DIR=/home/yuanhanyang.yhy/hllm-creator-baseline/phase2_smoke_outputs
mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES=0 python HLLM_Creator_eval_scripts/generate_usertitle.py \
  --ckpt_model_path "$MODEL_DIR/zero3_merge_states.pt" \
  --creative_tokenizer_path "$MODEL_DIR" \
  --data_path "$FAKE_DATA" \
  --output_path "$OUT_DIR/output_fake_limit2.parquet" \
  --batch_size 1 \
  --limit 2 \
  --max_tokens 32 \
  --config_file overall/LLM_deepspeed.yaml HLLM_Creator/HLLM_Creator.yaml \
  --MAX_ITEM_LIST_LENGTH 50 \
  --MAX_TEXT_LENGTH 64 \
  --item_pretrain_dir "$MODEL_DIR" \
  --user_pretrain_dir "$MODEL_DIR" \
  --creative_pretrain_dir "$MODEL_DIR" \
  --cluster_path "$MODEL_DIR/cluster_256.pt" \
  --use_ft_flash_attn False
```

Verify output:

```bash
python - <<'PY'
import pandas as pd
p = "phase2_smoke_outputs/output_fake_limit2.parquet"
df = pd.read_parquet(p)
print("rows", len(df))
print("columns", list(df.columns))
print(df[["llm_output"]].head().to_string())
PY
```

Expected:

```text
rows 2
llm_output column exists
```

If this passes, Phase 2 is successful.

## Step 10: Do Not Run Full Eval Yet

After the fake-data smoke test succeeds, stop and report:

- exact model directory
- exact fake data path
- whether `flash_attn` is installed
- whether generation output parquet exists
- GPU memory peak if available from `nvidia-smi`

Do not proceed to Amazon Electronics conversion or inference in Phase 2.

## Troubleshooting Notes

### If Hugging Face Download Uses Xet And Fails

Try:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
```

Then rerun the same download command.

### If `flash_attn` Blocks Import

Stop and report. The two possible fixes are:

1. install `flash_attn==2.5.9.post1`
2. patch `code/REC/model/HLLM/modeling_mistral.py` so `flash_attn.bert_padding`
   is optional, because HLLM-Creator imports Mistral even when a Llama/Qwen-style
   model is used

Do not choose one silently.

### If FAISS Is Missing

The official pretrained directory already has `cluster_256.pt`, so the Phase 2
generation smoke test should not need FAISS.

Only install FAISS if we need to run `cluster.py`.

### If `fbgemm_gpu` Is Missing

This is acceptable for HLLM-Creator. It is only required by the HSTU IDNet model
path.
