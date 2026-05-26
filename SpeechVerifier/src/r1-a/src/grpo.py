# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk

from open_r1.trainer import Qwen2AudioGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from sacrebleu import corpus_bleu
import re
import librosa

# os.environ['WANDB_MODE'] = 'offline'
# os.environ["WANDB_DIR"] = 'wandb'


# os.environ['WANDB_API_KEY'] = '07f2f7a5896a56faae572e0349fb940cd0c195f1'

def is_chinese(sentence):
    """
    判断句子是否为中文（包含中文字符）。

    :param sentence: 输入的句子（字符串）
    :return: 如果是中文返回 True，否则返回 False
    """
    return bool(re.search("[\u4e00-\u9fff]", sentence))


def clean_text(text):
    """
    清理句子，去除多余空格和特殊字符。

    :param text: 输入的句子（字符串）
    :return: 清理后的句子（字符串）
    """
    return re.sub(r"\s+", " ", text).strip()


def calculate_sacrebleu(candidate, reference):
    """
    使用 sacrebleu 计算候选句子和参考句子之间的 BLEU 分数，支持中文和英文。

    :param candidate: 候选句子（字符串或字符串列表）
    :param reference: 参考句子（字符串或字符串列表）
    :return: BLEU 分数（浮点数）
    """
    # 如果输入是单个句子，转换为列表
    if isinstance(candidate, str):
        candidate = [candidate]
    if isinstance(reference, str):
        reference = [reference]

    # 清理句子
    candidate = [clean_text(sentence) for sentence in candidate]
    reference = [clean_text(sentence) for sentence in reference]

    # 根据语言选择 tokenizer
    if is_chinese(candidate[0]) or is_chinese(reference[0]):
        tokenizer = "zh"  # 中文（包括中英文混合句子）
    else:
        tokenizer = "13a"  # 英文

    # 计算 BLEU 分数
    bleu_score = corpus_bleu(candidate, [reference], tokenize=tokenizer)
    return bleu_score.score


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, solution, task, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, tsk in zip(contents, solution, task):
        reward = 0.0

        # Extract answer from solution if it has think/answer tags
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

        # Extract answer from content if it has think/answer tags
        content_match = re.search(r'<answer>(.*?)</answer>', content)
        student_answer = content_match.group(1).strip() if content_match else content.strip()

        if tsk in ['gender_cla', 'pitch_cla', 'emotion_cla', 'speed_cla', 'energy_cla']:
            try:
                if ground_truth in student_answer:
                    reward = 1.0
            except Exception:
                pass
        else:
            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Compare the extracted answers
                    bleu = calculate_sacrebleu(student_answer, ground_truth)
                    if bleu < 20:
                        reward = 0.0
                    else:
                        reward = bleu / 50
                    # if student_answer == ground_truth:
                    #     reward = 2.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    try:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    except Exception as e:
        dataset = load_from_disk(script_args.dataset_name)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> {TASK_PROMPT} </answer> tags."

    def make_conversation_audio(example, sampling_rate=16000):
        audio_array = example["audio"]["array"]
        if example["audio"]["sampling_rate"] != sampling_rate:
            audio_array = librosa.resample(audio_array,
                                           orig_sr=example["audio"]["sampling_rate"],
                                           target_sr=sampling_rate)
        if example['task'] == 'emotion_cla':
            task_prompt = 'Speech emotion: [sad, happy, neutral, angry].'
        elif example['task'] == 'gender_cla':
            task_prompt = 'The speaker is [male, female].'
        elif example['task'] == 'pitch_cla':
            task_prompt = 'The pitch is [very low, low, medium, high, very high].'
        elif example['task'] == 'energy_cla':
            task_prompt = 'The speech energy is [very low, low,medium, high, very high].'
        elif example['task'] == 'speed_cla':
            task_prompt = 'The speech speed is [very slow, slow, medium, fast, vert fast].'
        else:
            task_prompt = ''
        return {
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": example['audio_id']},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["instruction"],
                                                                      TASK_PROMPT=task_prompt)},
                ],
                 },
            ],
            "wav": audio_array,
            'solution': '<answer> %s </answer>' % example["output"]
        }

    if "audio" in dataset[script_args.dataset_train_split].features:
        print("has audio in dataset")
        dataset = dataset.map(make_conversation_audio)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])
    else:
        print("no audio in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2AudioGRPOTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
