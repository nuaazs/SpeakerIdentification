import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Audio
import torchaudio
import soundfile as sf
import shutil
import os
import random
from os import listdir, getcwd
import sys
from tqdm import tqdm,trange
import logging
import pydub
from pydub import AudioSegment
import wave
import pandas as pd
import torchaudio
import random
from tqdm import tqdm
import pickle
import argparse
sys.path.append('/mnt/zhaosheng/brain/utils')

# SpeechBrain
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition

# Utils
from preprocess import audio_change,count_files,data_remove,data_cp,denoise_dict
from vad import vad_raw,vad_cat_ndarray,vad_kfth,vad_kfth_multi
from plot import plot_and_play,plot_and_play_list,plot_and_play_dir
from snr import SNR, SNR_power
from get_dataset import getCJKF,getCNCeleb,getVoxCeleb,getCNCeleb_dict,getCJKF_dict

# Plt Style
plt.rcParams['figure.figsize'] = (20,3)
plt.style.use('ggplot')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--span', type=int, default="5",help='')
parser.add_argument('--limit', type=float, default="0.8",help='')
parser.add_argument('--dataset', type=str, default="/mnt/cti_record_vad_data",help='')
args = parser.parse_args()


similarity = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)

vad_path = args.dataset
dataset_name = vad_path.split("/")[-1]
vad_wavs = getCNCeleb(vad_path)
vad_wavs_dict = getCNCeleb_dict(vad_path)


def testWav(wav,sr=8000,split_num=4,min_length=5,similarity=torch.nn.CosineSimilarity(dim=-1,eps=1e-6),similarity_limit=0.6):
    """
    质量检测
    """
    embedding_list = []
    wav_list = []
    
    if len(wav)/sr <= split_num*min_length:
        return False
    length = int(len(wav)/4)
    
    for index in range(split_num):
        tiny_wav = torch.tensor(wav[index*length:(index+1)*length]).unsqueeze(0)
        wav_list.append(tiny_wav)
        embedding_list.append(spkreg.encode_batch(tiny_wav)[0][0])    
    for embedding1 in embedding_list:
        for embedding2 in embedding_list:
            if similarity(embedding1, embedding2)<similarity_limit:
                return False
    return True





time_span = args.span

similarity_limit = args.limit
dotest = False
try:
    pre_f_read = open(f"8k_{dataset_name}_vad_5s_limit_{similarity_limit}.pkl", 'rb')
    pre_database = pickle.load(pre_f_read)
    pre_f_read.close()
    print("Load Success")
except:
    pre_database = {}
    dotest=True
used = []
pass_count = 0
#spkreg = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./pretrained_ecapa")

# 8k_预训练模型
spkreg = EncoderClassifier.from_hparams(
    source='/mnt/zhaosheng/brain/results/voxceleb12_ecapa_augment/22041901/save/CKPT+2022-04-19+14-43-59+00',
    hparams_file='/mnt/zhaosheng/brain/val_hyperparams.yaml')

database = {}
spk_num = 0


#similarity = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)

pbar = tqdm(vad_wavs_dict.keys())
for spk in pbar:
    if spk not in database.keys():
        for wav_file in vad_wavs_dict[spk]:
            file_name = wav_file.split("/")[-1]
            wav, sr = sf.read(wav_file)
            length = int(len(wav)/2)
            time_length = int(length/sr)
            
            if dotest:
                test_result = testWav(wav,similarity_limit=similarity_limit)
            else:

                if spk in pre_database.keys():
                    test_result = pre_database[spk]["test_result"]
                else:
                    test_result = False


            if not testWav(wav):
                pass_count+=1
                continue
            wav_1 = torch.tensor(wav[:length]).unsqueeze(0)
            wav_2 = torch.tensor(wav[length:]).unsqueeze(0)

            if time_span<= (time_length)/2:
                embedding_1 = spkreg.encode_batch(wav_1[:sr*time_span]).numpy()[0][0]
                embedding_2 = spkreg.encode_batch(wav_2[:sr*time_span]).numpy()[0][0]
            else:
                pass_count+=1
                continue


            database[spk] = {"spk_name":spk,"file_name":file_name,"embedding_1":embedding_1,"embedding_2":embedding_2,"test_result":test_result}   
            used.append(wav_file)
            spk_num+=1
            break
        pbar.set_description(f"TimeSpan:{time_span} Save:{spk_num} Pass:{pass_count}")

f_save = open(f"8k_{dataset_name}_vad_{time_span}s_limit_{similarity_limit}.pkl", 'wb')
pickle.dump(database, f_save)
f_save.close()
print(f"time_span:{time_span}:{spk_num}")










# from speechbrain.pretrained import EncoderClassifier
# from speechbrain.pretrained import SpeakerRecognition
# import torchaudio
# import random
# from tqdm import tqdm
# import pickle
# import time
# from multiprocessing.dummy import Pool as ThreadPool

# def make_data_base(spk,time_span,database,cn_wavs_dict):
#     #print(spk)
#     if spk not in database.keys():
#             for wav_file in cn_wavs_dict[spk]:
#                 file_name = wav_file.split("/")[-1]
#                 wav, sr = sf.read(wav_file)
#                 length = int(len(wav)/2)
#                 time_length = int(length/sr)
#                 wav_1 = torch.tensor(wav[:length]).unsqueeze(0)
#                 wav_2 = torch.tensor(wav[length:]).unsqueeze(0)
#                 if time_span<= time_length:
#                     embedding_1 = spkreg.encode_batch(wav_1[:sr*time_span]).numpy()[0][0]
#                     embedding_2 = spkreg.encode_batch(wav_2[:sr*time_span]).numpy()[0][0]
#                 else:
#                     continue
#                 database[spk] = {"spk_name":spk,"file_name":file_name,"embedding_1":embedding_1,"embedding_2":embedding_2}
#                 used.append(wav_file)
#                 break
    

# def process(item):
#     make_data_base(item,time_span,database,cn_wavs_dict)
    


# for time_span in [3,6,9]:
#     print(f"Start:{time_span}")
#     used = []
#     pass_count = 0
#     spkreg = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./pretrained_ecapa")
#     database = {}
#     testbase = {}
#     spk_num = 0

#     items = cn_wavs_dict.keys()
#     pool = ThreadPool()
#     pool.map(process, items)
#     pool.close()
#     pool.join()

#     print(len(database.keys()))

#     # 计算准确率
#     total_test_num = 0
#     top_num = 10
#     top=np.array([0]*top_num)
#     print(f"Time Span: {time_span}\nSpk Num: {len(database.keys())}")
#     for item in database.keys():
#         total_test_num += 1
#         results = []
#         embedding_1 = torch.tensor(database[item]["embedding_1"])
#         embedding_2 = torch.tensor(database[item]["embedding_2"])
#         name = database[item]["spk_name"]
#         print(f"Self test:{name}->   {similarity(embedding_1, embedding_2)}")
#         for base_item in database:
#             base_embedding_1 = torch.tensor(database[base_item]["embedding_1"])
#             results.append([similarity(embedding_2, base_embedding_1),base_item])
#         results = sorted(results,key=sort_result)
#         for index in range(5):
#             result = results[index]
#             if item in result:
#                 for ii in range(index,top_num):
#                     top[ii]+=1
#     print(top/total_test_num * 100)
#     f_save = open(f"database_1580_8000Hz_{time_span}s.pkl", 'wb')
#     pickle.dump(database, f_save)
#     f_save.close()
#     print(f"time_span:{time_span}:{spk_num}")