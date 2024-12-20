#! /bin/bash

models="
llama3.2:3b
llama3.1:8b
qwen2.5:3b
qwen2.5:7b
qwen2.5:14b
qwen2.5-coder:3b
qwen2.5-coder:7b
qwen2.5-coder:14b
deepseek-coder-v2:16b
yi-coder:9b
"

for model in $models; do
    if [ -f "data/HumanEval_samples_${model}_summary.json" ]; then
        echo "skip $model"
        continue
    fi
    echo "testing $model"
    python code_and_eval.py --model $model --use_ollama
done
