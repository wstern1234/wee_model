# 🧠 wee_model: Tiny LLM Fine-Tuning on Local Hardware

This repository contains a **locally fine-tuned GPT-2 (tiny)** model, trained for over **5 hours** on Windows using Python virtual environments, Hugging Face Transformers, and checkpoint management. I decided to name it "wee" model because it sounded Scottish (which I distantly am) and there are already so many tiny-models, mini-models, etc.

The goal was to prove that:
1. **You can fine-tune LLMs entirely on consumer hardware** — no GPUs in the cloud, no massive clusters.
2. **Checkpoints can be saved, restored, and reused** without retraining from scratch.
3. The resulting model can be **tested interactively** in a simple Python loop.

---

## ✨ Features
- ✅ Fine-tuned [`sshleifer/tiny-gpt2`](https://huggingface.co/sshleifer/tiny-gpt2) for domain-specific responses.
- ✅ Trained from scratch on a custom dataset for **18000 training steps**.
- ✅ Saved intermediate checkpoints for recovery and reproducibility.
- ✅ Interactive text generation loop for rapid testing.
- ✅ Fully reproducible inside a Python `venv` on Windows.

---

## 🛠️ Training Process
1. **Setup Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install transformers datasets torch
