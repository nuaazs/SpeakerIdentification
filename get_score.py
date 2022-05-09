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

TEST_NUM = 500
BLACK_LIMIT=0.7
LIMIT_INDEX = 0
import pickle
import random

npy_path = "/mnt/cjth_database_0427_16k_0.7"
# "/mnt/checked_cjth_vad_data_0423_span_10_allwav_and4segs"
embedding_1s = [os.path.join(npy_path,"embedding_1",filename) for filename in os.listdir(os.path.join(npy_path,"embedding_1"))]# if "diff" not in filename]
embedding_2s = [os.path.join(npy_path,"embedding_2",filename) for filename in os.listdir(os.path.join(npy_path,"embedding_2")) if "diff" in filename]
database = []
random.shuffle(embedding_2s)
inbase = embedding_2s[:1000]

#embedding_1s += inbase

outbase = embedding_2s[1000:]
# print(outbase[1])
outbase_names = set([item.split("/")[-1].split(".")[0] for item in outbase])
print(outbase_names)
# embedding_2s = outbase


print(f"Database:{len(embedding_1s)} Testbase{len(embedding_2s)}\n\tIn base:{len(inbase)}\n\tOut base:{len(outbase)}")

for embedding_1 in embedding_1s:
    spkname = embedding_1.split("/")[-1].split(".")[0]
    if spkname not in outbase_names:
        embedding = np.load(embedding_1)
        database.append([embedding,spkname])

testbase_in = []  
for embedding_2_in in inbase:
    spkname = embedding_2_in.split("/")[-1].split(".")[0]
    embedding = np.load(embedding_2_in)
    testbase_in.append([embedding,spkname])

    
testbase_out = []
for embedding_2_out in outbase:
    spkname = embedding_2_out.split("/")[-1].split(".")[0]
    embedding = np.load(embedding_2_out)
    testbase_out.append([embedding,spkname])
def sort_base(x):
    return (x[1])

database = sorted(database,key=sort_base)
testbase_in = sorted(testbase_in,key=sort_base)
testbase_out = sorted(testbase_out,key=sort_base)


# for index in range(len(database)):
#     assert database[index][1] == testbase[index][1] 
print(len(database),len(testbase_in),len(testbase_out))

def sort_result(x):
    return float(x[0])*(-1)


def getbest(a_list,b_list):
    return similarity(a_list, b_list)

similarity = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)

plt_info = []

total_test_num = 0
top_num = 10
top=np.array([0]*top_num)

for index,test_item in enumerate(testbase_in[:TEST_NUM]):
    
    results = []
    embedding_test = torch.tensor(test_item[0])
    name_test = test_item[1]
    print(f"Name:{name_test}")
    self_base_embedding = torch.tensor(database[index][0])
    total_test_num += 1
    for base_item in database:
        embedding_base = torch.tensor(base_item[0])
        name_base = base_item[1]
        results.append([getbest(embedding_test, embedding_base),name_base])
    results = sorted(results,key=sort_result)

    scores = [item[0] for item in results if item[0]<1]
    print(scores[LIMIT_INDEX])
    if  scores[LIMIT_INDEX]<BLACK_LIMIT:
        continue
    # Results
    for index in range(top_num):
        result = results[index]
        print(result)
        skp_name = result[1]
        score =  result[0]
        if name_test in result:
            for ii in range(index,top_num):
                top[ii]+=1

result = top/total_test_num * 100
result = [round(rst,2) for rst in result]
print(f"Test Number: {total_test_num}\nResults:{result}")
#plt_info.append([f"Limit={similarity_limit},Span={time_span}",top/total_test_num * 100])

total_test_num = 0
right_num = 0

for index,test_item in enumerate(testbase_out[:TEST_NUM]):
    results = []
    embedding_test = torch.tensor(test_item[0])
    name_test = test_item[1]
    # print(name_test)
    self_base_embedding = torch.tensor(database[index][0])
    self_test = getbest(embedding_test, self_base_embedding)
    total_test_num += 1
    for base_item in database:
        embedding_base = torch.tensor(base_item[0])
        name_base = base_item[1]
        results.append([getbest(embedding_test, embedding_base),name_base])
    results = sorted(results,key=sort_result)
    scores = [item[0] for item in results]
    # print(len(scores))
    max_score = scores[LIMIT_INDEX]
    # max_score = max_score//0.5

    print(max_score)
    if max_score<BLACK_LIMIT:
        right_num+=1
print(f"Open Set Test:{right_num/total_test_num*100.}%")
