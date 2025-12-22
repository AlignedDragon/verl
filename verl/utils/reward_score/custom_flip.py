# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import re
import json

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def tool_call_reward(predict_str: str) -> float:
    matches = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', predict_str, re.DOTALL)
    for match in matches:
        try:
            content = json.loads(match)
            # 2. Check logic on the parsed object
            if content.get("name") == "image_flip_tool" and "argument" in content:
                return 1.0
        except json.JSONDecodeError:
            continue # Skip malformed JSON
    return 0.0

def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    """
    Compute a reward.

    Reward is:
    - 1.0 if prediction matches ground truth
    - 0.0 otherwise 
    """
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str

    try:
        if answer==ground_truth:
            return 1.0
        elif ground_truth in answer:
            return 0.5
        else:
            return 0.0
    except Exception:
        return 0.0

def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1, tool_call_score: float = 0.1) -> float:
    acc_rwrd = acc_reward(predict_str, ground_truth, use_boxed)
    format_rwrd = format_reward(predict_str)
    tool_call_rwrd = tool_call_reward(predict_str)

    return (1.0 - format_score - tool_call_score)*acc_rwrd + format_score*format_rwrd + tool_call_score*tool_call_rwrd
