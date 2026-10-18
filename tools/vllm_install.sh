#!/usr/bin/env bash
# One-time setup on the GPU host: vLLM in its own venv (it pins its own torch), then start the server in tmux.
# Run detached: nohup bash ~/forte/tools/vllm_install.sh > ~/forte/logs/vllm_install.log 2>&1 &
set -euo pipefail
UV="$HOME/.local/bin/uv"
[ -x "$HOME/vllm-venv/bin/python" ] || "$UV" venv --python 3.12 "$HOME/vllm-venv"
"$UV" pip install --python "$HOME/vllm-venv/bin/python" vllm
"$HOME/vllm-venv/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
tmux has-session -t vllm 2>/dev/null || \
  tmux new-session -d -s vllm "bash $HOME/forte/tools/vllm_serve.sh 2>&1 | tee -a $HOME/forte/logs/vllm.log"
echo "server starting in tmux session vllm"
