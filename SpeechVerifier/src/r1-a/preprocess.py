from datasets import load_dataset
import soundfile as sf
import os

# 加载数据集
dataset = load_dataset("/mnt/private_hk/data/covost2_en-zh")  # 例如加载 Common Voice 数据集

# # 设置保存路径
# output_dir = "/apdcephfs_gy2/share_302533218/cedriccheng/data/R1_datasets/covost2_en_zh_test_v1_raw"
# os.makedirs(output_dir, exist_ok=True)


# 定义一个函数来修改每一行
def add_new_column(example, index):
    audio_data = example["context"]["array"]
    sample_rate = example["context"]["sampling_rate"]
    # question = example["instruction"]
    # answer = example["answer"]
    example['audio'] = audio_data
    example['problem'] = example["instruction"]
    example['solution'] = '<answer> %s </answer>' % example["answer"]
    # audio_filename = os.path.join(output_dir, f"audio_{index:05d}.wav")
    # sf.write(audio_filename, audio_data, sample_rate)
    # example['audio_url'] = audio_filename
    # with open(os.path.join(output_dir, f"audio_{index:05d}.txt"), "w") as f:
    #     f.write(example["answer"])
    return example


# 使用 map 函数应用修改
updated_dataset = dataset.map(add_new_column, with_indices=True)

# # 保存更新后的数据集（可选）
updated_dataset.save_to_disk("/apdcephfs_gy2/share_302533218/cedriccheng/data/R1_datasets/covost2_en-zh")

print("All files saved successfully.")
