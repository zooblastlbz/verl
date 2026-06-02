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

from verl.utils.reward_score.openmmreasoner import compute_score, extract_boxed_answer, format_reward


def test_extracts_rightmost_nested_boxed_answer():
    assert extract_boxed_answer(r"First \boxed{1}, finally \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_scores_correct_think_answer_response():
    result = compute_score(
        data_source="openmmreasoner",
        solution_str="<think>2 + 2 = 4</think><answer>4</answer>",
        ground_truth="4",
    )

    assert result["score"] == 1.0
    assert result["acc_score"] == 1.0
    assert result["format_reward_score"] == 1.0


def test_scores_format_only_when_answer_is_incorrect():
    result = compute_score(
        data_source="openmmreasoner",
        solution_str="<think>The answer is B.</think><answer>B</answer>",
        ground_truth="A",
    )

    assert result["score"] == 0.1
    assert result["acc_score"] == 0.0
    assert result["format_reward_score"] == 1.0


def test_accepts_trailing_number_without_format_reward():
    result = compute_score(
        data_source="openmmreasoner",
        solution_str="The final answer is 42.",
        ground_truth="42",
    )

    assert result["score"] == 0.9
    assert result["acc_score"] == 1.0
    assert result["format_reward_score"] == 0.0


def test_partial_format_reward_matches_official_recipe():
    response = "We calculate the expression 21 * 2 = 42 and therefore the final answer is 42."

    assert format_reward(response) == 0.8
