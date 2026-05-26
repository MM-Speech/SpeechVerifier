from datasets import Dataset as Dataset_

dataset = Dataset_.from_file("/home/limingze/F5-TTS/data/test_pinyin/raw.arrow")
print(dataset[-1])