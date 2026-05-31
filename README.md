<div align="center">

<h1>SpeechVerifier: RL-based Speech Verifier + FLOW-MPO for Nonverbal-Aware Speech Modeling</h1>

<p><strong>Unified repository for verifier alignment and flow-based preference optimization</strong></p>

<p>
  <a href="https://github.com/MM-Speech/SpeechVerifier">
    <img src="https://img.shields.io/badge/GitHub-MM--Speech%2FSpeechVerifier-black.svg" alt="Repo">
  </a>
</p>

</div>

`SpeechVerifier` combines two training tracks in one repository:

1. **SpeechVerifier (Verifier Alignment)**: RL-style alignment for speech verification tasks, built on top of the R1-V/R1-A style training recipe and implemented with **TRL `GRPOTrainer`**, targeting **Qwen2-Audio** style models.  
2. **FLOW-MPO (Flow Preference Optimization)**: A flow-based multi-objective preference optimization pipeline inspired by **F5-TTS**, with training code mainly under `src` and companion scripts for data generation, preference construction, and optimization.

This repo is intended for research on robust speech verification, nonverbal event sensitivity, and multi-dimensional preference optimization in speech generation/understanding systems.

---

## Highlights

- Dual-track framework: **RL verifier alignment** + **flow-based MPO** in one codebase.
- SpeechVerifier track adopts **GRPO** via TRL, following the practical recipe lineage of R1-V/R1-A.
- FLOW-MPO track supports **multi-dimensional preference supervision** rather than single-score preference learning.
- End-to-end workflow support: **data generation → preference selection → training → evaluation**.
- Practical integration for Qwen2-Audio style backbones and F5-TTS style flow training logic.

---

## Repository Overview

> The repository historically evolved from multiple codebases, so file layout may contain legacy scripts.
> Conceptually, it is organized into two main modules:

- **SpeechVerifier module**
  - RL/GRPO alignment for verifier behavior
  - Qwen2-Audio based policy training
  - Reward-driven optimization loop and evaluation utilities

- **FLOW-MPO module**
  - Flow-based training components (mainly in `src`)
  - Preference data generation and filtering pipeline
  - Multi-dimensional DPO/MPO style optimization scripts

If you are new to this repo, start from:
1. `SpeechVerifier` training entrypoints (GRPO configs/scripts)
2. `FLOW-MPO` user-facing pipeline scripts (usually clearer for data→train flow)
3. Shared config folders and `src` training modules

---

## Environment

Recommended environment (example):

```bash
conda create -n speechverifier python=3.10 -y
conda activate speechverifier
pip install -r requirements.txt
```

`setup.sh` installs editable package and required extras from the SpeechVerifier side.

## B) Flow-MPO environment

```bash
conda create -n flowmpo python=3.10 -y
conda activate flowmpo

cd Flow-MPO
pip install -e .
```

## Module 1: SpeechVerifier (GRPO / SFT)

## 1. Data format for GRPO (`SpeechVerifier/src/r1-a/src/grpo.py`)

Expected fields (audio branch):

- `audio` (HF audio field with `array` and `sampling_rate`)
- `audio_id`
- `instruction`
- `output`
- `task` (`emotion_cla`, `gender_cla`, `pitch_cla`, `energy_cla`, `speed_cla`, etc.)

`dataset_name` can be HF dataset name or local `load_from_disk` directory.

## 2. GRPO training (recommended entry)

```bash
cd SpeechVerifier/src/r1-a

export DEBUG_MODE=true
export LOG_PATH=./debug_log.txt
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12345 \
  src/grpo.py \
  --output_dir ./checkpoint/covost2_en_zh \
  --model_name_or_path /path/to/Qwen2-Audio-7B-Instruct \
  --dataset_name /path/to/your_dataset \
  --deepspeed local_scripts/zero3.json \
  --max_prompt_length 512 \
  --max_completion_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --logging_steps 1 \
  --bf16 \
  --report_to wandb \
  --gradient_checkpointing false \
  --attn_implementation flash_attention_2 \
  --num_train_epochs 10 \
  --save_steps 100 \
  --save_only_model true \
  --num_generations 8
```

## 3. SFT training

```bash
cd SpeechVerifier/src/r1-a

accelerate launch \
  --config_file configs/zero2.yaml \
  src/sft.py \
  --config configs/qwen2audio_sft_config.yaml
```

## 4. Evaluation

Example script:

- `SpeechVerifier/src/eval-a/test_qwen2audio_sr.py`
- `SpeechVerifier/src/eval-a/test_qwen2audio_st.py`

Before running, edit in-script constants:

- `dataset_name`
- `MODEL_PATH`
- `BSZ`

Run:

```bash
cd SpeechVerifier/src/eval-a
python test_qwen2audio_sr.py
```

## Module 2: FLOW-MPO (DPO workflow)

## 1. Prepare checkpoints and data

Recommended layout:

```text
Flow-MPO/
├── ckpts/
│   ├── F5TTS_Base/model_1200000.pt
│   ├── vocoder/
│   └── flow_dpo*/...
└── data/
    ├── librispeech_pc_test_clean_cross_sentence.lst
    ├── test-clean/...
    └── Emilia_ZH_EN_pinyin/vocab.txt
```

## 2. Generate candidate samples

```bash
cd Flow-MPO
bash user/prepare_dpo_pairs.sh
```

This launches multi-GPU generation via `user/gen_dpo_samples.py`.

## 3. Build preference pairs

```bash
cd Flow-MPO
python user/pick_dpo_pairs.py
```

This produces:

- `./user/dpo_data.json`

Important key names used by `DPODataset`:

- `winner_audio_path`
- `losser_audio_path` ← keep this exact spelling (code currently uses `losser`)

## 4. DPO training

```bash
cd Flow-MPO
bash user/dpo_training.sh
```

`user/dpo_training.sh` calls:

- `src/f5_tts/train/finetune_dpo.py`

Tune these first:

- `--pretrain`
- `--dataset_name`
- `--batch_size_per_gpu`
- `--grad_accumulation_steps`
- `--learning_rate`

## 5. Inference and eval

Single case inference:

```bash
cd Flow-MPO
python user/infer_one.py
```

Librispeech eval:

```bash
cd Flow-MPO
python user/eval_librispeech_test_clean.py \
  --eval_task wer \
  --lang en \
  --gen_wav_dir ./user/libri_test_clean_dpo \
  --librispeech_test_clean_path /path/to/test-clean \
  --gpu_nums 8
```

## Acknowledgement

This project builds upon open-source progress in:

- Qwen2-Audio style speech-language modeling
- F5-TTS style flow-matching TTS training
- DPO-style preference optimization and reward-aligned generation
- DeepSpeed / vLLM / Accelerate ecosystem for large-scale training and inference

