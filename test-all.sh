#! /bin/bash

for model in llama3.1-8b; do
    if [ -f "data/HumanEval_samples_${model}_summary.json" ]; then
        echo "skip $model"
        continue
    fi
    echo "testing $model"
    python code_and_eval.py --model $model | tee data/HumanEval_samples_${model}.log | grep processing
done


for model in llama3.2:3b llama3.1:8b qwen2.5:3b qwen2.5:7b qwen2.5:14b; do
    if [ -f "data/HumanEval_samples_${model}_summary.json" ]; then
        echo "skip $model"
        continue
    fi
    echo "testing $model"
    python code_and_eval.py --model $model --use_ollama | tee data/HumanEval_samples_${model}.log | grep processing
done
