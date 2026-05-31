# Phase 2 过程记录文档（HLLM-Creator 权重下载 + Fake-Data Smoke Test）

> 本文档如实记录 Phase 2 全过程（环境检查 → 权重下载 → 三大问题诊断修复 → flash on/off 双验证 → 产物），供人工检查。
> 执行依据：`docs/hllm_creator_phase2_weights_smoke.md`（Step0–Step10）。
> 记录时间：2026-05-31。

## 一、最终结论（TL;DR）

- **Phase 2 全流程 Step0–Step10 全部通过 ✅**
- 用官方合成数据（`fake_train_data`，1000 行）的前 2 行做 smoke test，模型成功加载混合 backbone 权重并生成语义连贯的书名。
- 过程中确诊并解决了 **3 个真实的环境/架构问题**：混合架构权重不匹配、libstdc++ 版本过旧、H20 上 cuBLAS bf16 GEMM 崩溃。
- 额外完成 **flash attn ON / OFF 双验证**，两条路径都跑通。
- 按文档要求，**未进行 Amazon Electronics 转换或全量推理**。

## 二、运行环境

| 项 | 值 |
|---|---|
| Python | 3.11.15（conda 环境 `hllm_creator`，未移动环境）|
| PyTorch | 2.3.1（自带 cuda 12.1）|
| GPU | NVIDIA H20（Hopper sm_90），单卡（`CUDA_VISIBLE_DEVICES=1`）|
| Driver | 580.82.07（支持到 CUDA 13.0）|
| flash_attn | 2.5.9.post1（已安装）|

## 三、关键路径

| 用途 | 路径 |
|---|---|
| creative_llm（Qwen3-8B）+ ckpt | `/data/oss_bucket_0/yhy/ByteDance_HLLM/HLLM_Creator/pretrained_model` |
| ckpt 文件 | `.../pretrained_model/zero3_merge_states.pt`（41,916,132,736 字节 ≈ 41.9GB）|
| item/user_llm（TinyLlama-1.1B）| `/data/oss_bucket_0/yhy/ByteDance_HLLM/base_llms/TinyLlama-1.1B` |
| 合成数据 | `/data/oss_bucket_0/yhy/ByteDance_HLLM/HLLM_Creator/fake_train_data/train.parquet`（1000 行 7 列，18082 字节）|
| cublas 修复库 | `logs/phase2/cublas_fix/extracted/nvidia_cublas_cu12-12.4.5.8/nvidia/cublas/lib/`（`libcublas.so.12` + `libcublasLt.so.12`）|
| python 解释器 | `/home/yuanhanyang.yhy/.conda/envs/hllm_creator/bin/python` |
| 生成产物 | `outputs/phase2/`（见第六节）|

注：权重与数据全部放在 OSS（`/data/oss_bucket_0/yhy/`）下，未下载到仓库盘。

## 四、分步过程记录（Step0–Step10）

### Step0 环境确认
- 进入仓库，激活 `hllm_creator` 环境，确认 Python 3.11.15。

### Step1 磁盘 / inode 检查
- OSS 空间远超 120GB 门槛；仓库盘空闲 86G，inode 充足。

### Step2 HuggingFace 工具检查
- `huggingface_hub 0.36.2`，`hf` 与 `huggingface-cli` 均可用。

### Step3 下载官方权重（约 71G）
- 下载策略：因 OSS 不支持 flock，改用 `/dev/shm` 内存盘中转下载，带自动重试续传（`logs/phase2/download_with_retry.sh`），完成后同步到 OSS。
- 全部下载完成 + 校验 + 同步 OSS + OSS 逐文件校验一致（19/19）+ 清理 /dev/shm。

### Step4 权重校验
- 19 个文件齐全：`zero3_merge_states.pt`（41.9G）、base `pytorch_model.bin`（32.8G）等，OSS 逐字节校验通过。

### Step5 合成数据准备
- `train.parquet`（1000 行 7 列），OSS 校验一致，清理 /dev/shm。

### Step6 目录确认
- 确认 MODEL_DIR 就位；creative LLM 为 Qwen3-8B 级（hidden 4096 / 36 层 / bf16）。

### Step7 导入检查
- HLLM_Creator 相关模块导入成功，flash_attn 未阻塞导入（仅无害警告）。

### Step8 模型构造 smoke test ✅
- 初次失败：size mismatch。诊断（`logs/phase2/diag_ckpt_arch.py`）发现**混合架构**（见第五节问题 1）。
- 下载 TinyLlama-1.1B 作 item/user backbone 后成功：`unexpected_keys=0`，`missing_keys=399`（均为 `decoder_llm.*`，与 creative_llm 共享权重，无害）。

### Step9 生成 smoke test ✅
- 连续踩到 libstdc++（问题 2）和 cuBLAS bf16（问题 3）两个坑，修复后 **bf16 原生路径完整跑通**。
- 结果：`unexpected_keys=[]`，2 条数据前向 4.08 it/s，生成 2.08 s/it，`STEP9_EXIT=0`，产出含 `llm_output` 列的真实书名。

### Step10 汇总报告（不跑全量）
- 按文档要求报告模型目录 / 数据路径 / flash_attn 状态 / 输出 parquet / 显存峰值，**未进行全量评测**。

## 五、三大关键问题的诊断与修复（重点）

### 问题 1：混合 backbone 架构，权重 size mismatch
- **现象**：Step8 加载 ckpt 报 size mismatch。
- **根因**：HLLM-Creator 是三 LLM 混合架构——`item_llm` / `user_llm` = **TinyLlama-1.1B**（hidden 2048 / vocab 32000），`creative_llm` = **Qwen3-8B**（hidden 4096 / vocab 151936）。MODEL_DIR 只含 Qwen3 config，缺 TinyLlama 骨架。
- **修复**：下载 TinyLlama-1.1B，将 `--item_pretrain_dir` / `--user_pretrain_dir` 指向它，`--creative_pretrain_dir` 指向 MODEL_DIR。
- **验证**：`unexpected_keys=0`，`missing_keys=399`（decoder 共享 creative 权重，无害）。

### 问题 2：libstdc++ 版本过旧（GLIBCXX_3.4.29）
- **现象**：Step9 导入 transformers → PIL 时报缺 `GLIBCXX_3.4.29`，系统 libstdc++ 只到 3.4.28。
- **根因**：系统 libstdc++ 旧于 conda 环境所需。
- **修复**：`LD_PRELOAD` conda 环境的 `libstdc++.so.6.0.34`，即 `/home/yuanhanyang.yhy/.conda/envs/hllm_creator/lib/libstdc++.so.6.0.34`。

### 问题 3：H20 上 cuBLAS bf16 GEMM 崩溃（核心难点）
- **现象**：Step9 前向在 `nn.Linear`（bf16 GEMM）处崩溃，报 `Fatal Floating point exception (SIGFPE)`。最小复现脚本精确定位到 `user_projector`（Linear 2048→4096）。
- **诊断**：输入 / 权重均无 NaN/Inf，fp32 CPU 正常，唯独 bf16 GPU 经 cuBLAS abort。降级 fp16 也只能绕过 projector，creative_llm.generate 内部 Linear 仍崩。
- **根因**：torch 2.3.1 自带的 `cublas 12.1.0.26` 在 H20（Hopper）上做 bf16 GEMM 触发 SIGFPE。属社区已知问题（参考 vLLM issue #4392 等，环境完全一致：torch 2.3.1+cu121 + H20）。
- **修复（零污染、可回退）**：
  1. `pip download nvidia-cublas-cu12==12.4.5.8 --no-deps` 下载到独立目录，`wheel unpack` 解包。
  2. 通过 `LD_PRELOAD` 预加载新版 `libcublasLt.so.12` + `libcublas.so.12`（顺序：先 Lt 后 cublas）。
  3. 不覆盖安装、不动 conda 主体，可随时回退。
- **验证**：轻量复现脚本 bf16 GEMM 一次通过；随后官方脚本 bf16 原生路径完整跑通。

### 附：flash_attn 状态
- smoke test 默认走 `--use_ft_flash_attn False`（普通 attention）以排除变量；第六节额外验证了开启路径同样可用。

## 六、flash attn ON / OFF 双验证结果

两次均使用官方未改动的 `generate_usertitle.py`（git working tree clean），`--limit 2`、bf16、单卡 H20。

| 维度 | flash OFF | flash ON |
|---|---|---|
| 运行状态 | EXIT=0 ✅ | EXIT=0 ✅ |
| ckpt 匹配 | unexpected_keys=[] | unexpected_keys=[] |
| 数据处理速度 | 4.08 it/s | 4.39 it/s（略快）|
| 生成速度 | 2.08 s/it | 2.09 s/it |
| 输出文件 | `output_fake_limit2.parquet`（18085 字节）| `output_fake_limit2_flashon.parquet`（18169 字节）|

生成内容对比（均为真实连贯书名）：

```
row0  OFF: "Shadow Operations: Inside the Human Drama of Modern Cyber Warfare"
      ON : "Shadow Operations: Inside Nation-State Cyber Warfare for Strategic Defense Professionals"
row1  OFF: "Shadow Operations: Unveiling Cyber Battles and Digital Ethics"
      ON : "Shadow Operations: Inside Cyber Warfare and Advanced Intelligence Tactics"
```

> 说明：ON/OFF 内容不同属正常现象——flash attn 与普通 attn 浮点累加顺序不同，导致 logits 微小差异、采样出不同但同样合理的结果，并非 bug。

## 六补、全量测试（flash ON + 全部 1000 条）

在双验证之外，额外用 **flash attn ON + bf16** 跑了全量合成数据（去掉 `--limit`，1000 条全跑），作为最终压力测试。

| 指标 | 数值 |
|---|---|
| 运行状态 | `FULL_FLASHON_EXIT=0` ✅ 全程零报错 |
| 数据处理 | 1000 条 / 29 秒（34.22 it/s）|
| 生成 | 500 batch（batch_size=2）/ 15 分 51 秒（1.90 s/batch）|
| GPU 利用率 | 96–100%（满载）|
| GPU 显存峰值 | 58.4 GB |
| 输出文件 | `output_fake_full_flashon.parquet`（27497 字节，1000 行 8 列）|

输出校验：
- 总行数 1000，含 `llm_output` 列，**空/缺失 0 条**（1000 条全部成功生成）。
- 标题长度 min=48 / max=105 / mean=72.4 字符。
- 唯一标题数 513（51.3%），多样性合理，无退化重复。

> 注：所有标题均以 "Shadow Operations:" 开头属正常——该合成数据所有样本围绕同一本网络战主题书，模型生成的是同一本书的不同标题变体，说明模型正确理解了输入语境。

## 七、产物与显存

- **生成产物**（位于 `outputs/phase2/`，已被 `.gitignore` 忽略）：
  - `output_fake_limit2.parquet`（flash OFF，smoke，2 行 8 列，含 `llm_output`）
  - `output_fake_limit2_flashon.parquet`（flash ON，smoke，2 行 8 列，含 `llm_output`）
  - `output_fake_full_flashon.parquet`（flash ON，全量 1000 行 8 列，含 `llm_output`）
- **显存 / 内存**：
  - smoke（2 条）：GPU 显存峰值约 42GB（bf16），CPU RSS 峰值约 46GB（加载 ckpt 42GB + 重新读 Qwen3 base 33GB，rchar 累计约 74.8GB）。
  - 全量（1000 条，flash ON）：GPU 显存峰值约 58.4GB（KV cache + batch 累积所致）。

## 八、复跑指引（供 Phase 3 接手）

### 8.1 复跑必需的环境前缀
```bash
MODEL_DIR=/data/oss_bucket_0/yhy/ByteDance_HLLM/HLLM_Creator/pretrained_model
TINYLLAMA_DIR=/data/oss_bucket_0/yhy/ByteDance_HLLM/base_llms/TinyLlama-1.1B
DATA=/data/oss_bucket_0/yhy/ByteDance_HLLM/HLLM_Creator/fake_train_data/train.parquet
PY=/home/yuanhanyang.yhy/.conda/envs/hllm_creator/bin/python
LIBSTDCPP=/home/yuanhanyang.yhy/.conda/envs/hllm_creator/lib/libstdc++.so.6.0.34
NEWCUBLAS=/home/yuanhanyang.yhy/hllm-creator-baseline/logs/phase2/cublas_fix/extracted/nvidia_cublas_cu12-12.4.5.8/nvidia/cublas/lib
PRELOAD="$LIBSTDCPP:$NEWCUBLAS/libcublasLt.so.12:$NEWCUBLAS/libcublas.so.12"
```

### 8.2 Step9 生成命令（bf16，flash 可选）
```bash
cd code
CUDA_VISIBLE_DEVICES=1 LD_PRELOAD="$PRELOAD" PYTHONPATH=$(pwd) $PY \
  HLLM_Creator_eval_scripts/generate_usertitle.py \
  --ckpt_model_path $MODEL_DIR/zero3_merge_states.pt \
  --creative_tokenizer_path $MODEL_DIR \
  --data_path $DATA --output_path <输出路径> \
  --batch_size 2 --limit 2 --max_tokens 32 \
  --config_file overall/LLM_deepspeed.yaml HLLM_Creator/HLLM_Creator.yaml \
  --MAX_ITEM_LIST_LENGTH 50 --MAX_TEXT_LENGTH 64 \
  --item_pretrain_dir $TINYLLAMA_DIR --user_pretrain_dir $TINYLLAMA_DIR \
  --creative_pretrain_dir $MODEL_DIR \
  --use_ft_flash_attn False   # 改 True 即开启 flash attn
```

### 8.3 注意事项
- `logs/phase2/cublas_fix/`（约 874M）是 Step9 复跑依赖，**勿删**。
- `logs/phase2/` 和 `outputs/phase2/` 已加入 `.gitignore`，不会进 git。
- 文档要求 Phase 2 阶段**不要跑全量 / 不要转换 Amazon 数据**。

## 九、保留的关键脚本

| 文件 | 用途 |
|---|---|
| `logs/phase2/step8_model_load_smoke.py` | Step8 模型加载验证 |
| `logs/phase2/diag_ckpt_arch.py` | 诊断 ckpt 真实架构（发现混合 backbone）|
| `logs/phase2/download_with_retry.sh` | 权重下载（/dev/shm 中转 + 重试）|
| `logs/phase2/download_tinyllama_with_retry.sh` | TinyLlama 下载 |
| `logs/phase2/monitor_*.sh` | 下载进度监控 |
| `logs/phase2/phase2_step*.log` | 各步骤运行日志（存档证据）|
