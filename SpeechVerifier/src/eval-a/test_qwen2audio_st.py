from transformers import Qwen2AudioForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
from datasets import load_dataset, load_from_disk
import librosa
from vllm import LLM, SamplingParams
from sacrebleu import corpus_bleu
import soundfile as sf

# dataset_name = '/apdcephfs_gy2/share_302533218/cedriccheng/data/R1_datasets/covost2_en-zh'
dataset_name = '/mnt/private_hk/data/covost2_zh_en_test'
# MODEL_PATH = "/mnt/private_hk/project/R1-V/src/r1-a/checkpoint/covost2_en-zh/checkpoint-400"  # Qwen2vl-2b-Instruct for original scores
MODEL_PATH = '/mnt/private_hk/data/Qwen2-Audio-7B-Instruct'
BSZ = 64  # reduce it if GPU OOM
OUTPUT_PATH = f"./results/{dataset_name}.json"


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

QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> {TASK_PROMPT} </answer> tags."

messages = []

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

try:
    dataset = load_dataset(dataset_name)
except Exception as e:
    dataset = load_from_disk(dataset_name)

all_outputs = []  # List to store all answers

BLEU = []
pbar = tqdm(dataset['test'], total=len(dataset['test']))
res={}
con=0
for example in pbar:
    con+=1

    # if example['task'] == 'emotion_cla':
    #     task_prompt = 'Speech emotion: [sad, happy, neutral, angry].'
    # elif example['task'] == 'gender_cla':
    #     task_prompt = 'The speaker is [male, female].'
    # elif example['task'] == 'pitch_cla':
    #     task_prompt = 'The pitch is [very low, low, medium, high, very high].'
    # elif example['task'] == 'energy_cla':
    #     task_prompt = 'The speech energy is [very low, low,medium, high, very high].'
    # elif example['task'] == 'speed_cla':
    #     task_prompt = 'The speech speed is [very slow, slow, medium, fast, vert fast].'
    # else:
    task_prompt = ''

    message = [
        # {"role": "system",
        #  "content": [
        #      {
        #          "type": "text",
        #          "text": SYSTEM_PROMPT
        #      }
        #  ]},
        {"role": "user",
         "content": [
             {
                 "type": "audio",
                 "audio_url": '1'
             },
             {
                 "type": "text",
                 # "text": QUESTION_TEMPLATE.format(Question=example["instruction"], TASK_PROMPT=task_prompt)
                 "text": example["instruction"]
             },
         ],
         }]

    text = [processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False, add_audio_id=True)]

    audio_array = example["context"]["array"]
    if example["context"]["sampling_rate"] != 16000:
        audio_array = librosa.resample(audio_array,
                                       orig_sr=example["context"]["sampling_rate"],
                                       target_sr=16000)

    sf.write(f'/mnt/private_hk/data/covost2_zh_en_test_raw/{con}.wav', audio_array, 16000)

    # inputs = processor(
    #     text=text,
    #     audios=[audio_array],
    #     return_tensors="pt",
    #     padding=True,
    #     padding_side="left",
    #     add_special_tokens=False,
    # )
    #
    # inputs = inputs.to("cuda")
    inputs = [
        {
            'prompt': text[i],
            'multi_modal_data': {
                'audio': audio_array
            }
        } for i in range(len(text))
    ]

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        # print()
        # print('=' * 40)
        # # print(f"Inputs[{i}]: {inputs[i]['prompt']!r}")
        # print(f"Generated text: {generated_text!r}")

        # Extract answer from content if it has think/answer tags
        content_match = re.search(r'<answer>(.*?)</answer>', generated_text)
        student_answer = content_match.group(1).strip() if content_match else generated_text.strip()
        bleu = calculate_sacrebleu(student_answer, example["answer"])
        BLEU.append(bleu)
        res.setdefault(con, {'pred': student_answer, 'gt': example["answer"], 'bleu': bleu})

        print(f'REF: {example["answer"]}\n'
              f'PRD: {student_answer}\n'
              f'BLEU: {bleu}\n'
              f'mean:{sum(BLEU) / len(BLEU)}\n')
        # pbar.set_postfix(bleu=sum(BLEU) / len(BLEU))
with open('./covost_zh_en_re.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)
