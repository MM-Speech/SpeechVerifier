from datasets import Dataset, Audio, load_dataset,DatasetDict
import pandas as pd
import numpy as np




prompts = ['Listen to this English audio and give me a smooth Chinese translation.', \
           'Please transcribe the English speech and translate it into natural Chinese. Make sure the translation sounds fluent and natural.', \
            'Transcribe and translate this English audio into Chinese.']

covost_data = load_dataset(path='/mnt/private_hk/data/covost2_en-zh')
train_df = covost_data['train'].to_pandas()
dev_df = covost_data['dev'].to_pandas()
test_df = covost_data['test'].to_pandas()
# 重新命令 'sentence' 列 和 'translation' 列
def process_df(df: pd.DataFrame):
    df.rename(columns={'sentence': 'input', 'translation': 'output'}, inplace=True)

    df['instruction'] = np.random.choice(prompts, size=len(df))

    df['audio_id'] = df['audio'].apply(lambda x: x['path'])
    df['dataset'] = 'covost2_en-zh'
    df['task'] = 's2tt'

print('translating data to pandas...')

process_df(train_df)
process_df(dev_df)
process_df(test_df)

print('Successfully processed data to pandas!')

print('translating data to datasets...')

train_hf = Dataset.from_pandas(train_df)
dev_hf = Dataset.from_pandas(dev_df)
test_hf = Dataset.from_pandas(test_df)

print('Successfully translated data to datasets!')

print('Casting audio column to Audio()...')

train_hf = train_hf.cast_column('audio', Audio(sampling_rate=16000)).select(range(30000))
dev_hf = dev_hf.cast_column('audio', Audio(sampling_rate=16000))
test_hf = test_hf.cast_column('audio', Audio(sampling_rate=16000))

print('Successfully casted audio column to Audio()!')

# 创建包含子集的 DatasetDict
dataset_dict = DatasetDict({
    'train': train_hf,
    'dev': dev_hf,
    'test': test_hf
})

# 保存整个 DatasetDict
save_path = '/apdcephfs_gy2/share_302533218/cedriccheng/data/R1_datasets/covost2_en-zh'
dataset_dict.save_to_disk(save_path)

# # 上传数据集
# train_hf.push_to_hub('MYJOKERML/covost2_en-zh', split='train')
# dev_hf.push_to_hub('MYJOKERML/covost2_en-zh', split='dev')
# test_hf.push_to_hub('MYJOKERML/covost2_en-zh', split='test')