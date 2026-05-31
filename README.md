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
