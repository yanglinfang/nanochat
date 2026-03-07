#!/bin/bash

# nanochat training on RTX 3080 Ti (12GB VRAM)
# This is a LEARNING run — the model will be small and imperfect.
# The point is to watch the training loop, observe loss curves, and build intuition.
#
# Run via Docker:
#   docker compose -f docker/docker-compose.yml run train bash runs/run3080ti.sh
#
# Or run stages one-by-one (recommended for learning):
#   docker compose -f docker/docker-compose.yml run train bash
#   Then copy-paste each section below into the terminal.

set -e

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p $NANOCHAT_BASE_DIR
WANDB_RUN=dummy

echo "============================================"
echo "  nanochat Learning Lab — RTX 3080 Ti"
echo "============================================"

# -----------------------------------------------
# Stage 1: Prepare data and tokenizer (~2 min)
# -----------------------------------------------
echo ""
echo ">>> Stage 1: Downloading data and training tokenizer..."
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# -----------------------------------------------
# Stage 2: Pre-train a small model (~1-2 hours)
# -----------------------------------------------
# depth=6  → 6 transformer layers (vs 20-24 in full run)
# head-dim=64 → smaller attention heads
# max-seq-len=512 → shorter context (vs 2048)
# device-batch-size=2 → fits in 12GB VRAM (reduce to 1 if OOM)
# num-iterations=2000 → enough to see the loss curve shape
#
# EXPERIMENT IDEAS:
#   - Change --depth to 2 or 12, compare loss curves
#   - Set --matrix-lr=1.0 and watch loss explode (then set to 1e-8, watch it stall)
#   - Reduce --num-iterations to 200 for a quick test
echo ""
echo ">>> Stage 2: Pre-training (depth=6, ~1-2 hours)..."
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=2 \
    --total-batch-size=4096 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=200 \
    --num-iterations=2000 \
    --run=$WANDB_RUN

# Evaluate the base model
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

# -----------------------------------------------
# Stage 3: SFT — teach it to chat (~30 min)
# -----------------------------------------------
# This is the same concept as our Qwen LoRA fine-tuning in ClipOnAiML,
# except here we're doing full fine-tuning (all weights, no LoRA).
echo ""
echo ">>> Stage 3: SFT (teaching conversation)..."
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=2 \
    --total-batch-size=4096 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=$WANDB_RUN

# -----------------------------------------------
# Stage 4: Chat with your model!
# -----------------------------------------------
echo ""
echo ">>> Training complete! Chat with your model:"
echo "    python -m scripts.chat_cli -p 'What is the capital of France?'"
echo "    python -m scripts.chat_web  (for web UI on port 7860)"
