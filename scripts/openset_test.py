# Author: 𝕫𝕙𝕒𝕠𝕤𝕙𝕖𝕟𝕘
# Email : zhaosheng@nuaa.edu.cn
# Time  : 2022-05-07  16:49:14.000-05:00
# Desc  : OpenSet Test for Speacker indentification

import pickle
import random
import os
import numpy as np
from IPython import embed
import torch
from tqdm import tqdm
similarity = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)

BLACK_TH = 0.7
random.seed(19970218)

# Embedding Saving Fold
npy_path = "/mnt/cjth_database_0508_16k_0.7"

# From same wav
embedding_1s = [[os.path.join(npy_path,"embedding_1",filename),filename.split(".")[0]] for filename in os.listdir(os.path.join(npy_path,"embedding_1")) if "diff" not in filename]

# From diff wav
embedding_2s = [[os.path.join(npy_path,"embedding_2",filename),filename.split(".")[0]] for filename in os.listdir(os.path.join(npy_path,"embedding_2")) if "diff" in filename]


# Split embedding_2s to <inbase> and <outbase>
random.shuffle(embedding_2s)
names_inbase = [item[1] for item in embedding_1s]

inbase = embedding_2s[:1000]
names_inbase_append = [item[1] for item in inbase]

outbase = embedding_2s[1000:2000]
names_outbase = [item[1] for item in outbase]


base = embedding_1s
# embedding_2s = outbase

black_names = names_inbase + names_inbase_append
outblack_names = names_outbase

# embedding tensor, spk_id
blackbase = []

# embedding tensor, spk_id, inbase or not
testbase = []

for filename in os.listdir(os.path.join(npy_path,"embedding_1")):
    spk_id = filename.split(".")[0]
    embedding_npy_path = os.path.join(npy_path,"embedding_1",filename)
    
    if spk_id in black_names:
        embedding = np.load(embedding_npy_path)
        blackbase.append([embedding,spk_id,True])
    else:
        pass

for item in [filename for filename in os.listdir(os.path.join(npy_path,"embedding_2")) if "diff" in filename]:
    spk_id = item.split(".")[0]
    embedding_npy_path = os.path.join(npy_path,"embedding_2",item)
    if spk_id in black_names:
        embedding = np.load(embedding_npy_path)
        testbase.append([embedding,spk_id,True])
    elif spk_id in outblack_names:
        embedding = np.load(embedding_npy_path)
        testbase.append([embedding,spk_id,False])


testbase = sorted(testbase,key=lambda x:x[1])
blackbase = sorted(blackbase,key=lambda x:x[1])
print(f"BlackBase Name length:{len(black_names)} OutBase Name length:{len(outblack_names)}")
print(f"TestBase:{len(testbase)} BlackBase:{len(blackbase)}")
# embed()





plt_info = []

total_test_num = 1
total_test_num_for_top = 1
top_num = 10
top=np.array([0]*top_num)
tp,fn,fp,tn = 0,0,0,0
pbar = tqdm(testbase)

for test_item in pbar:
    desc=f"TP:{tp/total_test_num:.2f},FN:{fn/total_test_num:.2f},FP:{fp/total_test_num:.2f},TN:{tn/total_test_num:.2f},Top1:{top[0]/total_test_num_for_top*100:.2f}%,Top2:{top[1]/total_test_num_for_top*100:.2f}%,Top3:{top[2]/total_test_num_for_top*100:.2f}%,Top4:{top[3]/total_test_num_for_top*100:.2f}%,Top5:{top[4]/total_test_num_for_top*100:.2f}%"
    desc+=f"Top6:{top[5]/total_test_num_for_top*100:.2f}%,Top7:{top[6]/total_test_num_for_top*100:.2f}%,Top8:{top[7]/total_test_num_for_top*100:.2f}%,Top9:{top[8]/total_test_num_for_top*100:.2f}%,Top10:{top[9]/total_test_num_for_top*100:.2f}%"
    pbar.set_description(desc)
    results = []
    embedding_test = torch.tensor(test_item[0])
    name_test = test_item[1]


    total_test_num += 1
    for base_item in blackbase:
        embedding_base = torch.tensor(base_item[0])
        name_base = base_item[1]
        results.append([similarity(embedding_test, embedding_base),name_base])
    results = sorted(results,key=lambda x:x[0]*-1)
    max_score = results[0][0]
    isblack = True if (max_score >= BLACK_TH) else False
    inbase = test_item[2]
    print(f"Score: {max_score} -> isblack? {isblack} | inBase? {inbase}")

    
    if isblack == inbase:
        if isblack:
            tp+=1
            total_test_num_for_top+=1
        else:
            tn+=1
            continue
    else:
        if isblack:
            fp+=1
            total_test_num_for_top+=1
        else:
            fn+=1
            continue

    # Results
    for index in range(top_num):
        result = results[index]
        #print(result)
        skp_name = result[1]
        score =  result[0]
        if name_test in result:
            
            for ii in range(index,top_num):
                top[ii]+=1
result = top/total_test_num * 100
result = [round(rst,2) for rst in result]
print(result)
embed()
