import os
import sys

sys.path.append(os.getcwd())

import json
from importlib.resources import files
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from datasets.arrow_writer import ArrowWriter


def main():
    result = []
    new_result = []
    duration_list = []
    text_vocab_set = set()

    with open(meta_info, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            uttr, text, norm_text = line.split("|")
            norm_text = norm_text.strip()
            wav_path = Path(dataset_dir) / "wavs" / f"{uttr}.wav"
            duration = sf.info(wav_path).duration
            if duration < 0.4 or duration > 30:
                continue
            result.append({"audio_path": str(wav_path), "text": norm_text, "duration": duration})
            # result.append({"audio_path": str(wav_path)})
            duration_list.append(duration)
            text_vocab_set.update(list(norm_text))

        # result.append({"test": 0})
        with open("/home/limingze/F5-TTS/data/all_dpo_data.txt", "r") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                splits = line.strip().split('\t')
                winner_audio = splits[0]
                loser_audio = splits[1]
                text = splits[2]
            
                winner_duration = sf.info(Path(winner_audio)).duration
                loser_duration = sf.info(Path(loser_audio)).duration

                winner_audio = winner_audio.replace("/home/limingze/F5-TTS/src/f5_tts/infer_out", "/mnt/yuyin1/limingze/F5-TTS/data")
                loser_audio = loser_audio.replace("/home/limingze/F5-TTS/src/f5_tts/infer_out", "/mnt/yuyin1/limingze/F5-TTS/data")
                # result.append({"winner_audio_path": winner_audio, "loser_audio_path": loser_audio, "text": text, "winner_duration": winner_duration, "loser_duration": loser_duration})
    # with open("/home/limingze/F5-TTS/data/all_dpo_data.txt", "r") as f:
    #     lines = f.readlines()
    #     for line in tqdm(lines):
    #         splits = line.strip().split('\t')
    #         winner_audio = splits[0]
    #         loser_audio = splits[1]
    #         text = splits[2]
        
    #         winner_duration = sf.info(Path(winner_audio)).duration
    #         loser_duration = sf.info(Path(loser_audio)).duration

    #         winner_audio = winner_audio.replace("/home/limingze/F5-TTS/src/f5_tts/infer_out", "/mnt/yuyin1/limingze/F5-TTS/data")
    #         loser_audio = loser_audio.replace("/home/limingze/F5-TTS/src/f5_tts/infer_out", "/mnt/yuyin1/limingze/F5-TTS/data")
    #         new_result.append({"winner_audio_path": winner_audio, "loser_audio_path": loser_audio, "text": text, "winner_duration": winner_duration, "loser_duration": loser_duration})
    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        print(len(result))
        print(result[-1])
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    # add alphabets and symbols (optional, if plan to ft on de/fr etc.)
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list)/3600:.2f} hours")


if __name__ == "__main__":
    tokenizer = "pinyin"  # "pinyin" | "char"

    dataset_dir = "/home/limingze/F5-TTS/data/LJSpeech-1.1"
    dataset_name = f"test_{tokenizer}"
    meta_info = os.path.join(dataset_dir, "metadata.csv")
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")

    main()
