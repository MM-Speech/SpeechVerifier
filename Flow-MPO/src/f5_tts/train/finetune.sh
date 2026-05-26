accelerate launch --mixed_precision=fp16 --main_process_port=25566 src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_Base \
    --dataset_name LibriTTS \
    --batch_size_per_gpu 8 \
    --max_samples 32 \
    --finetune \
    --pretrain ./ckpts/F5TTS_Base/model_1200000.pt \
    --tokenizer pinyin 