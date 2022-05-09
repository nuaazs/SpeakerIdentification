import os
import numpy as np
import matplotlib.pyplot as plt
def getVoxCeleb(raw_data_root):
    #raw_data_root = "/mnt/nyh/dataset1/wav/"
    speackers = sorted([os.path.join(raw_data_root,_file) for _file in os.listdir(raw_data_root) if _file[0].isdigit() or _file.startswith("id")])
    
    all_wavs = []
    for speacker in speackers:
        for tag in os.listdir(speacker):
            tag_path = os.path.join(speacker,tag)
            wavs = [os.path.join(tag_path,wav_file) for wav_file in os.listdir(tag_path)]
            all_wavs+=wavs
    print(f"Wavs Num:{len(all_wavs)}")
    return all_wavs

def getCNCeleb(raw_data_root,maxnum=None):
    #raw_data_root = "/mnt/nyh/dataset1/wav/"
    speackers = sorted([os.path.join(raw_data_root,_file) for _file in os.listdir(raw_data_root) if "id" in _file or (_file[0].isdigit())])
    
    if maxnum:
        if len(speackers) >= maxnum:
                speackers = speackers[:maxnum]
    print(f"Speacker Num:{len(speackers)}")
    all_wavs = []
    for speacker in speackers:
        wavs = [os.path.join(speacker,wav_file) for wav_file in os.listdir(speacker) if "wav" in wav_file and ".prof" not in wav_file]
        all_wavs+=wavs
        if maxnum:
            if len(all_wavs) >= maxnum:
                break
    print(f"Wavs Num:{len(all_wavs)}")
    return all_wavs

def getCNCeleb_dict(raw_data_root,maxnum=None,plot=False):
    distribution = np.zeros(10,)
    _dict = {}
    #raw_data_root = "/mnt/nyh/dataset1/wav/"
    speackers = sorted([_file for _file in os.listdir(raw_data_root) if "id" in _file or (_file[0].isdigit())])
    if maxnum:
        if len(speackers) >= maxnum:
                speackers = speackers[:maxnum]

    all_wavs_num = 0
    for speacker in speackers:
        speacker_path = os.path.join(raw_data_root,speacker)
        wavs = [os.path.join(speacker_path,wav_file) for wav_file in os.listdir(speacker_path) if "wav" in wav_file and ".prof" not in wav_file]
        _dict[speacker] = wavs
        wav_numbers = min(len(wavs),9)
        distribution[wav_numbers]+=1
        all_wavs_num += len(wavs)
        
    if plot:
        plt.figure()
        plt.title(f"Wavs Num:{all_wavs_num} Speacker Num:{len(speackers)}")
        plt.xlabel("number of wavs")
        plt.ylabel("number of speaker")
        plt.bar(np.arange(10),distribution)
        plt.show()
    else:
        print(f"Speacker Num:{len(speackers)}")
        print(f"Wavs Num:{all_wavs_num}")

    return _dict

def getCJKF(raw_data_root):
    #raw_data_root = "/datacjkf/wav_bak/"
    speackers = sorted([os.path.join(raw_data_root,_file) for _file in os.listdir(raw_data_root) if _file[0].isdigit() and "wav" not in _file])
    print(f"Speacker Num:{len(speackers)}")
    
    all_wavs = []
    for speacker in speackers:
        speackers_name = speacker.split("/")[-1]
        for tag in [tag for tag in os.listdir(speacker) if "wav" not in tag]:
            tag_path = os.path.join(speacker,tag)
            wavs = [os.path.join(tag_path,wav_file) for wav_file in os.listdir(tag_path)]
            all_wavs+=wavs
    print(f"Wavs Num:{len(all_wavs)}")
    return all_wavs


def getCJKF_dict(raw_data_root,plot=False):
    distribution = np.zeros(10,)
    _dict = {}
    speackers = sorted([os.path.join(raw_data_root,_file) for _file in os.listdir(raw_data_root) if _file[0].isdigit() and "wav" not in _file])
    all_wavs_num = 0
    for speacker in speackers:
        speackers_name = speacker.split("/")[-1]
        tags = [tag for tag in os.listdir(speacker) if "wav" not in tag]
        #print(tags)
        wavs = []
        for tag in tags:
            tag_path = os.path.join(speacker,tag)
            wavs += [os.path.join(tag_path,wav_file) for wav_file in os.listdir(tag_path)]
        _dict[speackers_name] = wavs
        wav_numbers = min(len(wavs),9)
        distribution[wav_numbers]+=1
        all_wavs_num+=len(wavs)
    if plot:
        plt.figure()
        plt.title(f"Wavs Num:{all_wavs_num} Speacker Num:{len(speackers)}")
        plt.xlabel("number of wavs")
        plt.ylabel("number of speaker")
        plt.bar(np.arange(10),distribution)
        plt.show()
    else:
        print(f"Speacker Num:{len(speackers)}")
        print(f"Wavs Num:{all_wavs_num}")
    return _dict