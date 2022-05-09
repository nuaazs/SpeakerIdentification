### Make Test Files for Dataset-CJKF 8k


raw_data_root = "/datacjkf/wav_vad/"
import shutil
import os
from os import listdir, getcwd
import random
data_dict = {}
all_wavs=[]
wavs_count = 0
speackers = sorted([os.path.join(raw_data_root,_file) for _file in os.listdir(raw_data_root) if _file[0].isdigit()])

print(f"Speacker Num:{len(speackers)}")
all_wavs = []
for speacker in speackers:
    speacker_id = speacker.split("/")[-1]
    #print(speacker_id)
    if speacker_id not in data_dict.keys():
        data_dict[speacker_id]=[]
    
    for tag in os.listdir(speacker):
        tag_path = os.path.join(speacker,tag)
        wavs = [os.path.join(tag_path,wav_file) for wav_file in os.listdir(tag_path) if ".wav" in wav_file]
        data_dict[speacker_id]+=wavs
        all_wavs+=wavs
        #print(len(wavs))
        wavs_count += len(wavs)
    
    #print(f"Speacker {speacker_id} Wavs Num:{len(data[speacker_id])}")
    #break
print(f"Wavs Num:{wavs_count}")



speacker_list = data_dict.keys()
distribution = np.zeros((100,))
largest_wav_num = 0
for speacker in speacker_list:
#     if speacker== "18338927560":
#         break
    wav_num = len(data_dict[speacker])
    if largest_wav_num<wav_num:
        largest_wav_num = wav_num
        largest_speacker = speacker
    distribution[wav_num]+=1
plt.bar(np.arange(100),distribution)
largest_wav_num



root = "/datacjkf/wav_vad/"
speacker_paths = [os.path.join(root,sid) for sid in os.listdir(root)]
print(speacker_paths)
for path in speacker_paths:
    for root, dirs, files in os.walk(path, topdown=False):
        print(dirs)
        print(files)
        if not files and not dirs:
            os.rmdir(root)
            

            
data_dict_len = {}
for specker in data_dict.keys():
    num = len(data_dict[specker])
    if str(num) not in data_dict_len:
        data_dict_len[str(num)]=[]
    data_dict_len[str(num)].append(specker)
    
    
root = "/datacjkf/wav_vad/"
speckers_2 = data_dict_len["2"]
speckers_num = len(speckers_2)
print(speckers_num)
NUM = 200
_string = ""
# same
for xx in range(int(NUM/2)):
    random_int = random.randint(0,speckers_num-1)
    path_wav_1,path_wav_2 = data_dict[speckers_2[random_int]]
    path_wav_1 = path_wav_1.replace("/datacjkf/wav_vad/","")
    path_wav_2 = path_wav_2.replace("/datacjkf/wav_vad/","")
    _string += f"\n1 {path_wav_1} {path_wav_2}"


# diff
for xx in range(int(NUM/2)):
    random_int = random.randint(0,speckers_num-1)
    while True:
        random_int2 = random.randint(0,speckers_num-1)
        if random_int2 != random_int:
            break

    path_wav_1 = data_dict[speckers_2[random_int]][0]
    path_wav_2 = data_dict[speckers_2[random_int2]][0]
    path_wav_1 = path_wav_1.replace("/datacjkf/wav_vad/","")
    path_wav_2 = path_wav_2.replace("/datacjkf/wav_vad/","")
    _string += f"\n0 {path_wav_1} {path_wav_2}"
print(_string)