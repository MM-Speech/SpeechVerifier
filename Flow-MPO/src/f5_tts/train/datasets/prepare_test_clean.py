import os
from tqdm import tqdm
import random
def read_txt_files(directory):
    # 创建一个空列表来存储文件名和内容
    file_contents = []
    ref_audio_list = []
    ref_text_list = []
    gen_text_list = []
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".original.txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 将文件名和内容添加到列表中
                audio_path = file.replace("original.txt","wav")
                audio_path = f"{root}/{audio_path}"
                ref_audio_list.append(audio_path)
                ref_text_list.append(content)
                gen_text_list.append(content)
                # file_contents.append(f"{root}/{audio_path}\t{content}\n")
    # gen_text_list = ref_text_list        
    random.shuffle(gen_text_list)
    for i in range(len(ref_audio_list)):
        file_contents.append(f"{ref_audio_list[i]}\t{ref_text_list[i]}\t{gen_text_list[i]}\n")
    return file_contents

def write_to_output(file_contents, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(file_contents):
            f.write(item)

if __name__ == "__main__":
    # 指定要读取的文件夹路径
    directory = "/home/limingze/F5-TTS/data/LibriTTS"
    
    # 指定输出文件的路径
    output_file = "/home/limingze/F5-TTS/data/libritts_test_clean_new.txt"
    
    # 读取所有txt文件的内容
    file_contents = read_txt_files(directory)
    
    # 将内容写入输出文件
    write_to_output(file_contents, output_file)
    
    print(f"所有txt文件的内容已整理并写入到 {output_file}")