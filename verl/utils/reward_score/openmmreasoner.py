# Copyright 2026 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenMMReasoner-compatible reward function.

The scoring recipe follows:
https://github.com/EvolvingLMMs-Lab/OpenMMReasoner/blob/main/custom_rewards/lmms_lab_recipe.py
"""

import os
import re
from typing import Any

_CHOICES = tuple("ABCDEFGH")
_FORMAT_WEIGHT = 0.1

_JUDGE_PROMPT = """You are a strict evaluator assessing answer correctness.
You must output 1 for fully correct answers and 0 for any other case.
# Input

Ground Truth Answer:

```
{answer}
```

Model Prediction:

```
{prediction}
```
# Evaluation Rules

- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer,
  it can be directly in option letters or the content of the options.
- For open-ended questions:
 * Score 1 if the prediction matches the answer semantically, it can be in different format.
 * Score 0 for partially correct answers or answers with extra incorrect information,
   even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct
# Strict Output format

1 or 0"""

_JUDGE_PROMPT_WITH_QUESTION = """You are a strict evaluator assessing answer correctness.
You must output 1 for fully correct answers and 0 for any other case.
You will receive the question, the ground truth answer, and the model prediction.
# Input

Question:

```
{question}
```

Ground Truth Answer:

```
{answer}
```

Model Prediction:

```
{prediction}
```
# Evaluation Rules

- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer,
  it can be directly in option letters or the content of the options.
- For open-ended questions:
 * Score 1 if the prediction matches the answer semantically, it can be in different format.
 * Score 0 for partially correct answers or answers with extra incorrect information,
   even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct
# Strict Output format

1 or 0"""


def extract_boxed_answer(predict_str: str) -> str:
    """Extract the rightmost balanced ``\\boxed{...}`` answer."""
    boxed_start = "\\boxed{"
    results = []
    pos = 0
    while True:
        start_pos = predict_str.find(boxed_start, pos)
        if start_pos == -1:
            break
        pos = start_pos + 1
        brace_count = 0
        cursor = start_pos + len(boxed_start) - 1
        while cursor < len(predict_str):
            char = predict_str[cursor]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    results.append(predict_str[start_pos + len(boxed_start) : cursor])
                    break
            cursor += 1
    return results[-1] if results else ""


def extract_answer(predict_str: str) -> str:
    """Extract an answer tag, a boxed answer, or the last trailing number."""
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if answer_match:
        return answer_match.group(1)

    boxed_answer = extract_boxed_answer(predict_str)
    if boxed_answer:
        return boxed_answer

    for line in reversed(predict_str.strip().splitlines()):
        number_match = re.search(r"\b(\d+(?:\.\d+)?)\b(?:\s*\.?\s*$)", line)
        if number_match:
            return number_match.group(1)
    return ""


def format_reward(predict_str: str) -> float:
    """Reward the official reasoning format or common mathematical fallback."""
    think_answer_pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>", re.DOTALL)
    if re.fullmatch(think_answer_pattern, predict_str):
        return 1.0
    if extract_boxed_answer(predict_str):
        return 1.0
    if len(predict_str.strip()) > 50:
        has_math = bool(re.search(r"[=\+\-\*/\(\)\[\]\\]", predict_str))
        if has_math and extract_answer(predict_str):
            return 0.8
    return 0.0


def simple_parse(predict_str: str) -> str:
    """Remove the trailing period used by many short answers."""
    if predict_str.endswith("."):
        predict_str = predict_str[:-1]
    return predict_str.strip()


def parse_mcq(predict_str: str) -> str:
    """Extract an A-H choice using the formats accepted by OpenMMReasoner."""
    if not predict_str or not predict_str.strip():
        return ""

    response = predict_str.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.strip(char)
    response = f" {response} "
    candidates: list[tuple[str, int, str]] = []

    patterns = [
        ("parentheses", lambda choice: f"({choice})"),
        ("period", lambda choice: f"{choice}."),
        ("colon", lambda choice: f"{choice}:"),
        ("right_paren", lambda choice: f"{choice})"),
        ("space", lambda choice: f"{choice} "),
        ("dash", lambda choice: f"{choice}-"),
        ("underscore", lambda choice: f"{choice}_"),
        ("equals", lambda choice: f"{choice}="),
    ]
    for pattern_name, render in patterns:
        for choice in _CHOICES:
            token = render(choice)
            if token in response:
                candidates.append((choice, response.rfind(token), pattern_name))

    answer_phrases = [
        "the answer is",
        "answer is",
        "the correct answer is",
        "correct answer is",
        "the answer",
        "answer",
        "correct answer",
        "the correct answer",
        "the best answer is",
        "best answer is",
        "the best answer",
        "best answer",
        "the option is",
        "option is",
        "the correct option is",
        "correct option is",
        "the choice is",
        "choice is",
        "the correct choice is",
        "correct choice is",
        "i choose",
        "i select",
        "i pick",
        "my answer is",
        "my choice is",
    ]
    lower_response = response.lower()
    for phrase in answer_phrases:
        if phrase in lower_response:
            phrase_start = lower_response.find(phrase)
            for choice in _CHOICES:
                choice_pos = response.find(choice, phrase_start)
                if choice_pos != -1:
                    candidates.append((choice, choice_pos, "phrase"))

    for choice in _CHOICES:
        if response.strip().startswith(choice):
            candidates.append((choice, 0, "start"))
        if response.strip().endswith(choice):
            candidates.append((choice, len(response) - 1, "end"))

    for index, choice in enumerate(_CHOICES, start=1):
        token = f"{index}. {choice}"
        if token in response:
            candidates.append((choice, response.rfind(token), "numbered"))

    if not candidates:
        for choice in _CHOICES:
            if choice in response:
                candidates.append((choice, response.rfind(choice), "fallback"))

    if not candidates:
        return ""

    priority = {
        "start": 10,
        "end": 9,
        "numbered": 8,
        "phrase": 7,
        "parentheses": 6,
        "period": 5,
        "colon": 4,
        "right_paren": 3,
        "space": 2,
        "dash": 1,
        "underscore": 1,
        "equals": 1,
        "fallback": 0,
    }
    candidates.sort(key=lambda candidate: (priority[candidate[2]], -candidate[1]), reverse=True)
    return candidates[0][0]


def relax_exact_match(predict_str: str, ground_truth: str, relax_portion: float = 0.9) -> float:
    """Apply the OpenMMReasoner relaxed exact-match rule."""
    if ground_truth in _CHOICES:
        return 1.0 if parse_mcq(predict_str) == ground_truth else 0.0
    if predict_str in ground_truth and len(predict_str) >= relax_portion * len(ground_truth):
        return 1.0
    if ground_truth in predict_str and len(ground_truth) >= relax_portion * len(predict_str):
        return 1.0
    return 1.0 if predict_str.strip() == ground_truth.strip() else 0.0


def _math_verify(predict_str: str, ground_truth: str) -> float:
    try:
        from math_verify import parse, verify

        return float(bool(verify(parse(ground_truth), parse(predict_str))))
    except Exception:
        return 0.0


def _llm_as_judge(predict_str: str, ground_truth: str, extra_info: dict[str, Any] | None) -> int:
    from openai import OpenAI

    if extra_info is not None and "question" in extra_info:
        prompt = _JUDGE_PROMPT_WITH_QUESTION.format(
            question=extra_info["question"], answer=ground_truth, prediction=predict_str
        )
    else:
        prompt = _JUDGE_PROMPT.format(answer=ground_truth, prediction=predict_str)

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        max_tokens=5,
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    )
    try:
        return int(response.choices[0].message.content)
    except Exception:
        return 0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    sandbox_fusion_url: str | None = None,
    concurrent_semaphore: Any = None,
) -> dict[str, Any]:
    """Return correctness, format, and combined OpenMMReasoner rewards."""
    del data_source, sandbox_fusion_url, concurrent_semaphore

    format_reward_score = format_reward(solution_str)
    predict_str = simple_parse(extract_answer(solution_str).strip())
    parsed_ground_truth = simple_parse(ground_truth)
    acc_score = relax_exact_match(predict_str, parsed_ground_truth)
    if acc_score == 0.0:
        acc_score = _math_verify(predict_str, parsed_ground_truth)

    use_llm_judge = os.getenv("USE_LLM_JUDGE", "False") == "True"
    if acc_score == 0.0 and use_llm_judge:
        acc_score = _llm_as_judge(predict_str, ground_truth, extra_info)
    if acc_score == 0.0 and use_llm_judge and format_reward_score == 0.0 and len(solution_str) < 500:
        acc_score = _llm_as_judge(solution_str, ground_truth, extra_info)

    score = (1.0 - _FORMAT_WEIGHT) * acc_score + _FORMAT_WEIGHT * format_reward_score
    return {
        "score": score,
        "acc_score": acc_score,
        "format_reward_score": format_reward_score,
        "predict_str": predict_str,
        "ground_truth": parsed_ground_truth,
    }
