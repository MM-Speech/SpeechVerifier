for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i nohup python user/gen_dpo_samples.py > ./output_$i.log 2>&1 &
done
