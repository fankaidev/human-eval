import fire
import json
from typing import Any
import openai
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness
import os

litellm = openai.OpenAI(api_key="sk-6264", base_url="https://llm.fankai.site")
ollama = openai.OpenAI(base_url="http://localhost:11434")


def complete_code(body: str, model: str, use_ollama: bool = False) -> str:
    start_anchor = "### start of completion"
    end_anchor = "### end of completion"
    prompt = f"""You are a profession python developer.
You will be given a python function signature with docstring, and you should finish the function body.

## input
```
{body}
```

## output
```python
{{ original part of method }}
    {start_anchor}
    {{ completion }}
    {end_anchor}
```
"""
    llm = ollama if use_ollama else litellm
    response = llm.chat.completions.create(
        messages=[{"content": prompt, "role": "user"}],
        model=model,
    )
    res: str = response.choices[0].message.content
    print(f"------\nanswer:\n{res}")
    if start_anchor in res and end_anchor in res:
        start_idx: int = res.find(start_anchor) + len(start_anchor)
        end_idx = res.find(end_anchor)
        return res[start_idx:end_idx]
    else:
        start_idx = res.rfind('"""') + 3
        return res[start_idx:].rstrip().rstrip("```")


def code_and_eval(model: str = "llama3.3-70b", limit: int = 0, use_ollama: bool = False):
    problems: dict[str, Any] = read_problems()
    output_file = f"data/HumanEval_samples_{model.replace('/', '-')}"
    with open(output_file + ".jsonl", "w") as fout:
        for idx, problem in enumerate(problems.values()):
            if limit > 0 and idx >= limit:
                print("======\nskipping", problem["task_id"])
                code = "    return None"
            else:
                print(f"======\nprocessing {problem['task_id']}\n{problem['prompt']}")
                code = complete_code(problem["prompt"], model, use_ollama)
            ans = {
                "task_id": problem["task_id"],
                "prompt": problem["prompt"],
                "completion": code,
            }
            fout.write(json.dumps(ans) + "\n")

    results = evaluate_functional_correctness(output_file, n_workers=10)
    print(results)
    with open(output_file + "_summary.json", "w") as fout:
        json.dump(results, fout)


if __name__ == "__main__":
    fire.Fire(code_and_eval)
