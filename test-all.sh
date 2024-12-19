#! /bin/bash

for model in llama3.2:3b llama3.1:8b qwen2.5:3b qwen2.5:7b qwen2.5:14b qwen2.5-coder:3b qwen2.5-coder:7b; do
    if [ -f "data/HumanEval_samples_${model}_summary.json" ]; then
        echo "skip $model"
        continue
    fi
    echo "testing $model"
    python code_and_eval.py --model $model --use_ollama
done

for model in llama3.1-8b llama3.3-70b claude-3.5-haiku gpt-4o-mini gemini-2.0-flash gemini-1.5-pro gemma2-9b qwen-coder qwen-72b yi-large-turbo deepseek-coder; do
    if [ -f "data/HumanEval_samples_${model}_summary.json" ]; then
        echo "skip $model"
        continue
    fi
    echo "testing $model"
    python code_and_eval.py --model $model
done