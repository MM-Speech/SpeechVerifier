import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from omegaconf import OmegaConf
from tqdm import tqdm
from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT

model = "F5-TTS"
model_cls = DiT
model_cfg = "/home/limingze/F5-TTS/src/f5_tts/configs/F5TTS_Base_train.yaml"
model_cfg = OmegaConf.load(model_cfg).model.arch
ckpt_file = "/home/limingze/F5-TTS/ckpts/F5TTS_Base/model_1200000.pt"
vocoder_name = "vocos"
vocoder_local_path = "/home/limingze/F5-TTS/ckpts/vocos-mel-24khz"
vocab_file = "/home/limingze/F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt"
ema_model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file)
vocoder = load_vocoder(vocoder_name=vocoder_name, local_path=vocoder_local_path)

# inference process

librispeech_fp = "/home/limingze/F5-TTS/data/librispeech_pc_test_clean_cross_sentence.lst"
with open(librispeech_fp, 'r') as fp:
    lines = fp.readlines()

for line in tqdm(lines):
    data = line.strip().split('\t')
    ref_audio_splits = data[0].split('-')
    ref_audio = f"/home/limingze/F5-TTS/data/librispeech/LibriSpeech/test-clean/{ref_audio_splits[0]}/{ref_audio_splits[1]}/{data[0]}.flac"
    ref_text = data[2]
    gen_text = data[5]
    # print(ref_audio,ref_text)
    ref_audio_, ref_text_ = preprocess_ref_audio_text(ref_audio, ref_text)

    generated_audio_segments = []
    audio_segment, final_sample_rate, spectragram = infer_process(
        ref_audio_,
        ref_text_,
        gen_text,
        ema_model,
        vocoder,
        mel_spec_type=vocoder_name,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        fix_duration=fix_duration,
    )
    generated_audio_segments.append(audio_segment)
    wav_path = ref_audio.split('/')[-1]
    wave_path = f"/home/limingze/F5-TTS/src/f5_tts/infer_out/librispeech_test_clean_wav/{data[3]}.wav"
    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            print(f.name)

