import fire
import json
from typing import Any
import openai
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness
import time
import os

litellm = openai.OpenAI(api_key="sk-6264", base_url="https://llm.fankai.site")
ollama = openai.OpenAI(base_url="http://localhost:11434/v1")


def complete_code(body: str, model: str, use_ollama: bool = False) -> str:
    prompt = f'''You are a profession python developer.

You will be given incomplete python code snippet with one ore more functions, and you should complete the last function.

# Example

* below example is used to demonstrate the output format.

<example_input>
```python
def sum_of_two_numbers(a: int, b: int) -> int:
    """ Given two integers, return their sum.
    >>> sum_of_two_numbers(1, 2)
    3
    """

    # Below is implementation
```
</example_input>

<example_output>
```python
def sum_of_two_numbers(a: int, b: int) -> int:
    """ Given two integers, return their sum.
    >>> sum_of_two_numbers(1, 2)
    3
    """

    # Below is implementation
    return a + b
```
</example_output>

# Input
<input>
```python
{body}
    # Below is implementation
```
</input>w

# Output
follow the format of the example output
* output a single block of valid python code in markdown format, wrapped by "```python" and "```"
* input part should be included in the output, without any modification
* must include the "# Below is implementation" line before the implementation part.
* no other text or comments or examples are allowed in the output.
* do not include the xml tags in the output.

'''
    llm = ollama if use_ollama else litellm
    response = llm.chat.completions.create(
        messages=[{"content": prompt, "role": "user"}],
        model=model,
    )
    assert response.choices[0].message.content is not None
    res: str = response.choices[0].message.content.strip()
    print(f"------\nresponse:\n{res}")

    assert "```" in res, "response does not include ```"

    res_lines = [line for line in res.splitlines()]
    if not any(line.strip() == "# Below is implementation" for line in res_lines):
        raise ValueError("Could not find '# Below is implementation' line in response")
    last_input_idx = min(i for i, line in enumerate(res_lines) if line.strip() == "# Below is implementation")
    impl_lines = [line for line in res_lines[last_input_idx + 1 : -1] if line.strip() != "```"]
    res = "\n".join(impl_lines)
    return res


def code_and_eval(model: str = "llama3.3-70b", limit: int = 0, use_ollama: bool = False):
    problems: dict[str, Any] = read_problems()
    output_file = f"data/HumanEval_samples_{model.replace('/', '-')}.jsonl"
    existing_entries = set()
    if os.path.exists(output_file):
        with open(output_file) as fin:
            for line in fin:
                entry = json.loads(line)
                existing_entries.add(entry["task_id"])
    for idx, problem in enumerate(problems.values()):
        if limit > 0 and idx >= limit:
            print(f"======\n[{model}] limit reached, stop processing")
            return
        elif problem["task_id"] in existing_entries:
            print(f"======\n[{model}] skip processing", problem["task_id"])
            continue
        else:
            print(f"======\n[{model}] processing {problem['task_id']}\n{problem['prompt']}")
            code = ""
            for _ in range(5):
                try:
                    code = complete_code(problem["prompt"], model, use_ollama)
                    print(f"------\nanswer:\n{code}")
                    if code.strip() == "":
                        raise ValueError("empty code")
                    break
                except Exception as e:
                    print(f"error: {e}")
                    if not use_ollama:
                        print(f"[{model}] will retry in 30 seconds")
                        time.sleep(30)
            if not code:
                print(f"------\n[{model}] failed to get code for {problem['task_id']}")
                continue

        ans = {
            "task_id": problem["task_id"],
            "prompt": problem["prompt"],
            "completion": code,
        }
        with open(output_file, "a") as fout:
            fout.write(json.dumps(ans) + "\n")

    results = evaluate_functional_correctness(output_file, n_workers=10)
    print(results)
    with open(output_file.replace(".jsonl", "_summary.json"), "w") as fout:
        results["model"] = model
        json.dump(results, fout)


if __name__ == "__main__":
    fire.Fire(code_and_eval)
