import os
import subprocess
from tqdm import tqdm
import soundfile as sf
###################
### audio_change ##
###################
'''
函数：audio_change
作用：批量更改音频,包括音频位置、采样率、音频格式、通道数方面的更改。
输入：old_path -- 旧音频库路径，如'/mnt/data_unzip/VoxCeleb2/dev/aac'
      new_path -- 即将创建的音频库位置，如 '/mnt/data_process/VoxCeleb1_2_16k_wav/vox2'
      channels -- 期望通道数， 如 1 
      sr -- 期望采样率, 如 16000
      format -- 期望音频格式，如str('.m4a')
输出：无
注意：old_path下必须是三级目录，即old_path下， id_dir -> audio_dir -> audio
'''

def audio_change(old_path, new_path, channels, sr, format):
    if os.path.exists(new_path) == False:
        os.mkdir(new_path)
    id_list = os.listdir(old_path)   #  id 列表
    for id in id_list: 
        id_path = os.path.join(old_path, id)  # 某个人的路径
        audio_dir = os.listdir(id_path)  #  某个人下的文件夹列表
        new_id_path = os.path.join(new_path, id)
        os.mkdir(new_id_path)
        for dir in audio_dir:    # 某个人下的某个文件夹
            dir_path = os.path.join(old_path, id, dir)  # 某个人下的某个文件夹 的路径
            audio_list = os.listdir(dir_path)   # 某个人下的某个文件夹 内的音频列表
            new_dir_path = os.path.join(new_id_path, dir)
            os.mkdir(new_dir_path)
            for audio in audio_list:  
                audio_path = os.path.join(old_path, id, dir, audio)  #  某个人下的某个文件夹 内的 某条音频 位置
                audio_format = audio_path.split('.')[1]
                old_format = f'.{audio_format}'
                new_name = audio.replace(old_format,format)  # 取新名字
                new_audio_path = os.path.join(new_dir_path, new_name)  # 新音频位置
                cmd = f'ffmpeg -i {audio_path} -ac {channels} -ar {sr} {new_audio_path}'
                subprocess.call(cmd, shell=True)

### audio_change 使用示例 ###
# old_path = '/mnt/data_unzip/VoxCeleb2/aac'
# new_path = '/mnt/malixin/vox2_aac_test'
# channels = 1
# sr = 8000
# formatf = str('.wav')
# audio_change(old_path, new_path, channels, sr, formatf)


# 两层的
def audio_change_2(old_path, new_path, channels, sr, format):
    if os.path.exists(new_path) == False:
        os.mkdir(new_path)
    id_list = os.listdir(old_path)   #  id 列表
    for id in id_list: 
        id_path = os.path.join(old_path, id)  # 某个人的路径
        audio_dir = os.listdir(id_path)  #  某个人下的文件夹列表
        new_id_path = os.path.join(new_path, id)
        os.mkdir(new_id_path)
        for audio in audio_dir:    # 某个人下的某个文件夹
#             dir_path = os.path.join(old_path, id, dir)  # 某个人下的某个文件夹 的路径
#             audio_list = os.listdir(dir_path)   # 某个人下的某个文件夹 内的音频列表
#             new_dir_path = os.path.join(new_id_path, dir)
#             os.mkdir(new_dir_path)
#             for audio in audio_list:  
            audio_path = os.path.join(old_path, id, audio)  #  某个人下的某个文件夹 内的 某条音频 位置
            audio_format = audio_path.split('.')[1]
            old_format = f'.{audio_format}'
            new_name = audio.replace(old_format,format)  # 取新名字
            new_audio_path = os.path.join(new_id_path, new_name)  # 新音频位置
            cmd = f'ffmpeg -i {audio_path} -ac {channels} -ar {sr} {new_audio_path}'
            subprocess.call(cmd, shell=True)

###################
### count_files ###
###################
'''
函数：count_files
作用：查询dir下文件（夹）数量
输入：要查询的文件夹地址
输出：文件夹内文件（夹）数量
'''
def count_files(dir):
    file_ls = os.listdir(dir)
    return len(file_ls)

###################
### data_remove ###
###################
'''
函数：data_remove
作用：删除dir2路径下与dir1路径下文件（夹）名字相同的文件（夹）
'''
def data_remove(dir1, dir2):
    id_list = os.listdir(dir1)
    for id in id_list:
        id_path = os.parh.join(dir2, id)
        cmd = f'sudo rm -rf {id_path}'
        subprocess.call(cmd, shell=True)


###################
##### data_cp #####
###################
'''
函数：data_cp
作用：递归复制dir1路径下所有文件（夹）到dir2路径下
'''
def data_cp(dir1, dir2):
    id_list = os.listdir(dir1)
    for id in id_list:
        id_path = os.path.join(dir1, id)
        new_id_path = os.path.join(dir2, id)
        cmd = f'cp -r {id_path} {new_id_path}'
        subprocess.call(cmd, shell=True)

        



def denoise_dict(raw_data_dict,new_data_path):
    os.makedirs(new_data_path,exist_ok=True)
    for spk in tqdm(raw_data_dict.keys()):

        wav_file_paths = raw_data_dict[spk]
        for wav_file_path in wav_file_paths:

            wav,sr = sf.read(wav_file_path)
            filename = wav_file_path.split("/")[-1]
            father_path = os.path.join(new_data_path,f"{spk}")
            os.makedirs(father_path,exist_ok=True)

            save_path = os.path.join(father_path,filename)
            filename_nosuffix = filename.split(".")[-2]
            noise_save_path = os.path.join(father_path,f"noise_{filename_nosuffix}.prof")

            cmd1 = f"sox {wav_file_path} -n noiseprof {noise_save_path}"
            #print(cmd1)
            subprocess.call(cmd1, shell=True)
            cmd2 = f"sox {wav_file_path} {save_path} noisered {noise_save_path} 0.5"
            #print(cmd2)
            subprocess.call(cmd2, shell=True)
