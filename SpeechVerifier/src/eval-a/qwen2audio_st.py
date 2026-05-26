import os.path

from transformers import Qwen2AudioForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
from datasets import load_dataset, load_from_disk
import librosa
from vllm import LLM, SamplingParams

# dataset_name = '/apdcephfs_gy2/share_302533218/cedriccheng/data/R1_datasets/covost2_en-zh'
ds = 'covost2_en-zh'
dataset_name = f'/mnt/private_hk/data/{ds}'
# MODEL_PATH = "/mnt/private_hk/project/R1-V/src/r1-a/checkpoint/covost2_en-zh/checkpoint-400"  # Qwen2vl-2b-Instruct for original scores
MODEL_PATH = '/mnt/private_hk/data/Qwen2-Audio-7B-Instruct'
BSZ = 64  # reduce it if GPU OOM
OUTPUT_PATH = f"./results/{ds}.json"



messages = []

try:
    dataset = load_dataset(dataset_name)
except Exception as e:
    dataset = load_from_disk(dataset_name)


def add_audio_id(example, idx):
    example["audio_id"] = f"audio_{idx:08d}"  # 格式化为4位数字，如 audio_0001
    example['instruction'] = 'Please translate the given speech to chinese.'
    return example


save_path = f'/apdcephfs_gy2/share_302533218/cedriccheng/data/R1_datasets/{ds}'
if not os.path.exists(save_path):
    # 添加 with_indices=True 以获取索引
    updated_dataset = dataset.map(add_audio_id, with_indices=True)
    updated_dataset.save_to_disk(save_path)


#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
llm = LLM(
    model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.98,
    enforce_eager=True,  # Disable CUDA graph, force call forward in every decode step.
    limit_mm_per_prompt={"audio": 5},
)
sampling_params = SamplingParams(
    temperature=0.7, top_p=0.01, top_k=1, repetition_penalty=1.1, max_tokens=256,
    stop_token_ids=[],
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# all_outputs = []  # List to store all answers
#
# BLEU = []
# pbar = tqdm(updated_dataset['test'], total=len(updated_dataset['test']))
# res = {}
# for example in pbar:
#
#     message = [
#         {"role": "user",
#          "content": [
#              {
#                  "type": "audio",
#                  "audio_url": example['audio_id']
#              },
#              {
#                  "type": "text",
#                  "text": example['instruction']
#              },
#          ],
#          }]
#
#     text = [processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False, add_audio_id=True)]
#
#     audio_array = example["audio"]["array"]
#     if example["audio"]["sampling_rate"] != 16000:
#         audio_array = librosa.resample(audio_array,
#                                        orig_sr=example["audio"]["sampling_rate"],
#                                        target_sr=16000)
#
#     inputs = [
#         {
#             'prompt': text[i],
#             'multi_modal_data': {
#                 'audio': audio_array
#             }
#         } for i in range(len(text))
#     ]
#
#     outputs = llm.generate(inputs, sampling_params=sampling_params)
#
#     for i, output in enumerate(outputs):
#         generated_text = output.outputs[0].text
#
#         # Extract answer from content if it has think/answer tags
#         student_answer = generated_text.strip()
#
#         res.setdefault(example['audio_id'], {'pred': student_answer, 'gt': example["output"]})
#
#         # pbar.set_postfix(bleu=sum(BLEU) / len(BLEU))
# with open(f'./{ds}.json', 'w', encoding='utf-8') as f:
#     json.dump(res, f, indent=4, ensure_ascii=False)

batch_size = 16  # 根据GPU内存调整
res = {}

# 获取测试集数据
test_data = updated_dataset["test"]
total_samples = len(test_data)

# 进度条（按batch更新）
with tqdm(total=total_samples, desc="Processing Batches") as pbar:
    for batch_start in range(0, total_samples, batch_size):
        # 获取当前batch的数据范围
        batch_end = min(batch_start + batch_size, total_samples)
        batch = test_data.select(range(batch_start, batch_end))

        # 批量准备输入
        inputs_batch = []
        meta_batch = []  # 保存当前batch的元数据

        for example in batch:
            # 构造消息模板
            message = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": example['audio_id']},
                    {"type": "text", "text": example['instruction']}
                ]
            }]

            # 生成prompt文本
            text = processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
                add_audio_id=True
            )

            # 处理音频
            audio_array = example["audio"]["array"]
            if example["audio"]["sampling_rate"] != 16000:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=example["audio"]["sampling_rate"],
                    target_sr=16000
                )

            # 收集输入数据和元数据
            inputs_batch.append({
                "prompt": text,
                "multi_modal_data": {"audio": audio_array}
            })
            meta_batch.append({
                "audio_id": example["audio_id"],
                "gt_answer": example["output"]
            })

        # 批量推理
        outputs = llm.generate(inputs_batch, sampling_params=sampling_params)

        # 处理结果
        for output, meta in zip(outputs, meta_batch):
            generated_text = output.outputs[0].text.strip()
            res[meta["audio_id"]] = {
                "pred": generated_text,
                "gt": meta["gt_answer"]
            }

        # 更新进度条
        pbar.update(len(batch))

# 保存结果（与原代码相同）
with open(f'./{ds}.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)