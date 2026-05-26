import os
import sys
sys.path.append(os.getcwd())
import multiprocessing as mp
from glob import glob
import numpy as np

import math
import os
import random
import string
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL


def run_sim(args):
    rank, test_set, ckpt_dir = args
    device = f"cuda:{rank}"

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    sims = []
    for wav1, wav2 in tqdm(test_set):
        wav_fn = wav2
        wav1, sr1 = torchaudio.load(wav1)
        wav2, sr2 = torchaudio.load(wav2)

        resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
        resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
        wav1 = resample1(wav1)
        wav2 = resample2(wav2)

        if use_gpu:
            wav1 = wav1.cuda(device)
            wav2 = wav2.cuda(device)
        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        # print(f"VSim score between two audios: {sim:.4f} (-1.0, 1.0).")
        sims.append([wav_fn, sim])

    return sims



def get_librispeech_test(gen_wav_dir, n_gpus):
    test_set_ = []
    for wav_dir in glob(f'{gen_wav_dir}/*'):
        for wav_path in glob(f"{wav_dir}/gen_*.wav"):
            test_set_.append([wav_path.replace('gen', 'gt'), wav_path])

    num_jobs = n_gpus
    if num_jobs == 1:
        return [(0, test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((i, test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


if __name__ == '__main__':
    gen_wav_dir = './user/libri_test_clean'
    n_gpus = 8

    test_set = get_librispeech_test(gen_wav_dir, n_gpus)
    wavlm_ckpt_dir = "/mnt/bn/sa-ag-data/jiangziyue/seed-tts-eval/ckpt/wavlm/wavlm_large_finetune.pth"

    # --------------------------- SIM ---------------------------
    sims = []
    with mp.Pool(processes=n_gpus) as pool:
        args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
        results = pool.map(run_sim, args)
        for r in results:
            sims.extend(r)
    
    score_dict = {}
    for item in sims:
        wav_fn, score = item

        spk_name = wav_fn.split('/')[-1][4:].split('_')[0]
        fn_id = wav_fn.split('_')[-1][:-4]
        if spk_name not in score_dict:
            score_dict[spk_name] = []
        else:
            score_dict[spk_name].append([fn_id, score])
    
    res_list = []
    for spk_name in score_dict:
        score_dict[spk_name] = sorted(score_dict[spk_name], key=lambda x: x[1])
        with open(f'./user/libri_test_clean/{spk_name}/gen_text.txt', 'r') as f:
            gen_text = f.read()
        with open(f'./user/libri_test_clean/{spk_name}/ref_text.txt', 'r') as f:
            ref_text = f.read()
        print(score_dict[spk_name])
        if len(score_dict[spk_name]) > 5:
            res_list.append({
                'winner_audio_path': f'./user/libri_test_clean/{spk_name}/gen_{spk_name}_{score_dict[spk_name][-1][0]}.wav',
                'losser_audio_path': f'./user/libri_test_clean/{spk_name}/gen_{spk_name}_{score_dict[spk_name][0][0]}.wav', 
                'ref_audio_path': f'./user/libri_test_clean/{spk_name}/gen_{spk_name}_{score_dict[spk_name][-1][0]}.wav', 
                'gen_text': gen_text, 
                'ref_text': ref_text
            })
    
    import json
    with open("./user/dpo_data.json", "w", encoding="utf-8") as file:
        json.dump(res_list, file, ensure_ascii=False, indent=4)