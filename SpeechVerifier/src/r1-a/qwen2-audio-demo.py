import requests

from transformers import AutoTokenizer, AutoProcessor
from transformers.pipelines.audio_utils import ffmpeg_read

from vllm import LLM, SamplingParams
import librosa

MODEL_PATH = '/mnt/private_hk/data/Qwen2-Audio-7B-Instruct'

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def qwen2_audio_batch():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # conversation1 = [
    #     {"role": "user", "content": [
    #         {"type": "audio", "audio_url": "./audio/glass-breaking-151256.mp3"},
    #         {"type": "text", "text": "What's that sound? Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."},
    #     ]},
    #     {"role": "assistant", "content": "It is the sound of glass shattering."},
    #     {"role": "user", "content": [
    #         {"type": "audio", "audio_url": "./audio/f2641_0_throatclearing.wav"},
    #         {"type": "text", "text": "What can you hear? Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."},
    #     ]}
    # ]
    #
    conversation2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "./audio/1272-128104-0000.flac"},
            {"type": "text",
             "text": "What does the person say? Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."},
        ]},
    ]

    conversation4 = [{'content': [{'text': SYSTEM_PROMPT, 'type': 'text'}],
                       'role': 'system'},
                      {'content':
                           [{'audio_url': '/apdcephfs_gy2/share_302533218/cedriccheng/data/R1_datasets/covost2_en_zh_test_v1_raw/audio_06838.wav',
                             'text': None, 'type': 'audio'},
                            {'audio_url': None,
                             'text': 'Please translate the given speech to Chinese  Output the thinking processin <think> </think> and final answer in <answer> </answer> tags.',
                             'type': 'text'}],
                       'role': 'user'}]

    # conversation3 = [
    #     {"role": "user", "content": [
    #         {"type": "text", "text": "How to make a pizza? Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."},
    #     ]},
    # ]

    # conversations = [conversation1, conversation2, conversation3]
    conversations = [conversation4]

    text = [processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False, add_audio_id=True)
            for conversation in conversations]

    audios = []
    for conversation in conversations:
        audio_infos_vllm = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audio_infos_vllm.append(
                            librosa.load(ele['audio_url'],
                                         sr=processor.feature_extractor.sampling_rate)[0]
                        )
        audios.append(audio_infos_vllm)
    print(text)
    inputs = [
        {
            'prompt': text[i],
            'multi_modal_data': {
                'audio': audios[i]
            }
        } for i in range(len(conversations))
    ]
    return inputs


def main():
    inputs = qwen2_audio_batch()


    llm = LLM(
        model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.98,
        enforce_eager=True,  # Disable CUDA graph, force call forward in every decode step.
        limit_mm_per_prompt={"audio": 5},
    )
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.01, top_k=1, repetition_penalty=1.1, max_tokens=256,
        stop_token_ids=[],
    )

    print(f"{inputs=}")
    for _ in range(10):
        outputs = llm.generate(inputs, sampling_params=sampling_params)

        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            # print()
            print('=' * 40)
            # print(f"Inputs[{i}]: {inputs[i]['prompt']!r}")
            print(f"Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
