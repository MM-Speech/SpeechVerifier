# Evaluate with Librispeech test-clean, ~3s prompt to generate 4-10s audio (the way of valle/voicebox evaluation)

import argparse
import json
import os
import sys

sys.path.append(os.getcwd())

import multiprocessing as mp
from importlib.resources import files
from tqdm import tqdm

import numpy as np
from f5_tts.eval.utils_eval import (
    run_asr_wer,
    run_sim,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_task", type=str, default="wer", choices=["sim", "wer"])
    parser.add_argument("-l", "--lang", type=str, default="en")
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True)
    parser.add_argument("-p", "--librispeech_test_clean_path", type=str, required=True)
    parser.add_argument("-n", "--gpu_nums", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")
    return parser.parse_args()

def get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth=False):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        if eval_ground_truth:
            gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
            gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")
        else:
            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt + ".wav")):
                continue
                # raise FileNotFoundError(f"Generated wav not found: {gen_utt}")
            gen_wav = os.path.join(gen_wav_dir, gen_utt + ".wav")

        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set

def main():
    args = get_args()
    eval_task = args.eval_task
    lang = args.lang
    librispeech_test_clean_path = args.librispeech_test_clean_path  # test-clean path
    gen_wav_dir = args.gen_wav_dir
    metalst = "./data/librispeech_pc_test_clean_cross_sentence.lst"

    gpus = list(range(args.gpu_nums))
    test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path)

    ## In LibriSpeech, some speakers utilized varying voice characteristics for different characters in the book,
    ## leading to a low similarity for the ground truth in some cases.
    # test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth = True)  # eval ground truth

    local = args.local
    if local:  # use local custom checkpoint dir
        asr_ckpt_dir = "../checkpoints/Systran/faster-whisper-large-v3"
    else:
        asr_ckpt_dir = ""  # auto download to cache dir
    wavlm_ckpt_dir = "/mnt/bn/sa-ag-data/jiangziyue/seed-tts-eval/ckpt/wavlm/wavlm_large_finetune.pth"

    # --------------------------- WER ---------------------------

    if eval_task == "wer":
        wer_results = []
        wers = []

        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_asr_wer, args)
            for r in results:
                wer_results.extend(r)

        wer_result_path = f"{gen_wav_dir}/{lang}_wer_results.jsonl"
        with open(wer_result_path, "w") as f:
            for line in wer_results:
                wers.append(line["wer"])
                json_line = json.dumps(line, ensure_ascii=False)
                f.write(json_line + "\n")

        wer = round(np.mean(wers) * 100, 3)
        print(f"\nTotal {len(wers)} samples")
        print(f"WER      : {wer}%")
        print(f"Results have been saved to {wer_result_path}")

    # --------------------------- SIM ---------------------------

    if eval_task == "sim":
        sims = []
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_sim, args)
            for r in results:
                sims.extend(r)
        print(sims)
        sim = round(sum(sims) / len(sims), 3)
        print(f"\nTotal {len(sims)} samples")
        print(f"SIM      : {sim}")


if __name__ == "__main__":
    main()
