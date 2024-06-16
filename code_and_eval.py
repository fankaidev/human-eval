import fire
import json
from typing import Any
from litellm import completion  # type: ignore
from litellm.utils import ModelResponse
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness
import os


def complete_code(body: str, model: str) -> str:
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

    messages = [{"content": prompt, "role": "user"}]
    response = completion(model=model, messages=messages)
    assert isinstance(response, ModelResponse), "invalid response"
    res: str = response["choices"][0]["message"]["content"]
    print(f"------\nanswer:\n{res}")
    if start_anchor in res and end_anchor in res:
        start_idx: int = res.find(start_anchor) + len(start_anchor)
        end_idx = res.find(end_anchor)
        return res[start_idx:end_idx]
    else:
        start_idx = res.rfind('"""') + 3
        return res[start_idx:].rstrip().rstrip("```")


def code_and_eval(model: str = "groq/llama3-70b-8192", limit: int = 0):
    problems: dict[str, Any] = read_problems()
    output_file = f"data/HumanEval_samples_{model.replace('/', '-')}.jsonl"
    with open(output_file, "w") as fout:
        for idx, problem in enumerate(problems.values()):
            if limit > 0 and idx >= limit:
                print("======\nskipping", problem["task_id"])
                code = "    return None"
            else:
                print(f"======\nprocessing {problem['task_id']}\n{problem['prompt']}")
                code = complete_code(problem["prompt"], model)
            ans = {
                "task_id": problem["task_id"],
                "prompt": problem["prompt"],
                "completion": code,
            }
            fout.write(json.dumps(ans) + "\n")

    results = evaluate_functional_correctness(output_file, n_workers=10)
    print(results)


if __name__ == "__main__":
    fire.Fire(code_and_eval)
