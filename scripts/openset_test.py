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


random.seed(19970218)
all_results = ""
all_top = ""
for BLACK_TH in np.arange(0.65,0.95,0.01):# np.linspace(0.75, 0.9, 30, True):
# BLACK_TH = 0.825
    # Embedding Saving Fold
    npy_path = '/mnt/cjth_database_0508_16k_0.7_10'

    # From same wav
    embedding_1s = [[os.path.join(npy_path,"embedding_1",filename),filename.split(".")[0]] for filename in os.listdir(os.path.join(npy_path,"embedding_1")) if "diff" not in filename]

    # From diff wav
    embedding_2s = [[os.path.join(npy_path,"embedding_2",filename),filename.split(".")[0]] for filename in os.listdir(os.path.join(npy_path,"embedding_2")) if "diff" in filename]


    # Split embedding_2s to <inbase> and <outbase>
    random.shuffle(embedding_2s)
    names_inbase = [item[1] for item in embedding_1s]

    inbase = embedding_2s[:100]
    names_inbase_append = [item[1] for item in inbase]

    outbase = embedding_2s[100:200]
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
        # print(f"Score: {max_score} -> isblack? {isblack} | inBase? {inbase}")

        
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
    result = top/total_test_num_for_top * 100
    result = [round(rst,2) for rst in result]
    # TP
    TP = tp/total_test_num*100.
    TN = tn/total_test_num*100.
    FP = fp/total_test_num*100.
    FN = fn/total_test_num*100.
    ACC = (tn+tp)/(tn+tp+fn+fp)*100.
    PPV = tp/(tp+fp)*100.
    TPR = tp/(tp+fn)*100.
    TNR = tn/(tn+fp)*100.
    now_result = f"{BLACK_TH},{ACC},{PPV},{TPR},{TNR},{TP},{TN},{FP},{FN}\n"
    all_results += now_result
    print(now_result)
    print(f"Top1 - Top10: {result}")
    all_top += f"{BLACK_TH},{result[0]},{result[1]},{result[2]},{result[3]},{result[4]},{result[5]},{result[6]},{result[7]},{result[8]},{result[9]}\n"
    # print(f"""
    # =================================
    #         BLACK_TH = {BLACK_TH}
    # =================================
    # \tTP:{tp/total_test_num*100.:.2f}%
    # \tTN:{tn/total_test_num*100.:.2f}%
    # \tFP:{fp/total_test_num*100.:.2f}%
    # \tFN:{fn/total_test_num*100.:.2f}%
    # \t准确率 ACC:{(tn+tp)/(tn+tp+fn+fp)*100.:.2f}%
    # \t精确率 PPV:{tp/(tp+fp)*100.:.2f}%
    # \t灵敏度 TPR:{tp/(tp+fn)*100.:.2f}%
    # \t特异度 TNR:{tn/(tn+fp)*100.:.2f}%
    # =================================
    # """)


    
embed()




import numpy as np
import matplotlib.pyplot as plt
top_results = all_top.split("\n")
top_info = []
for top_result in [item for item in top_results if item != ""]:
    _list = top_result.split(",")
    top_info.append(np.array([item for item in _list if item != ""],dtype=float))
top_info = np.array(top_info)

acc_results = all_results.split("\n")
acc_info = []
for top_result in [item for item in acc_results if item != ""]:
    _list = top_result.split(",")
    acc_info.append(np.array([item for item in _list if item != ""],dtype=float))
acc_info = np.array(acc_info)
plt.style.use('seaborn-darkgrid')
plt.figure(dpi=100)
plt.plot(acc_info[:,-2],acc_info[:,-4],linewidth=3)
plt.title("ROC")
plt.legend()
plt.xlabel("False Postive Rate")
plt.ylabel("True Postive Rate")
plt.savefig("roc.png")
plt.show()

plt.figure(dpi=100)
plt.plot(acc_info[:,0],acc_info[:,3],linewidth=1,label="TPR")
plt.plot(acc_info[:,0],acc_info[:,4],linewidth=1,label="TNR")
plt.title("")
plt.legend()
plt.xlabel("Black Threshold")
plt.ylabel("%")
plt.savefig("tnr_tpr.png")
plt.show()


plt.figure(dpi=100)
plt.scatter(acc_info[:,0],acc_info[:,1],linewidth=1,label="ACC")
plt.plot(acc_info[:,0],acc_info[:,1],linewidth=1,label="ACC")
plt.title("")
plt.xlabel("Black Threshold")
plt.ylabel("ACC %")
plt.savefig("acc.png")
plt.show()


plt.figure(dpi=100)
plt.plot(top_info[:,0],top_info[:,1],linewidth=1,label="Top 1")
plt.plot(top_info[:,0],top_info[:,2],linewidth=1,label="Top 2")
plt.plot(top_info[:,0],top_info[:,3],linewidth=1,label="Top 3")
plt.plot(top_info[:,0],top_info[:,4],linewidth=1,label="Top 4")
plt.plot(top_info[:,0],top_info[:,5],linewidth=1,label="Top 5")
plt.title("")
plt.legend()
plt.xlabel("Black Threshold")
plt.ylabel("%")
plt.savefig("topn.png")
plt.show()


# BLACK_TH = 0.8
# TP:0.40,FN:0.10,FP:0.19,TN:0.30
# Top1:61.02%,Top2:63.56%,Top3:64.41%,Top4:65.25%,Top5:65.25%Top6:65.25%,Top7:65.25%,Top8:65.25%,Top9:65.25%,Top10:65.25%: 




# BLACK_TH = 0.85


# =================================
#         BLACK_TH = 0.825
# =================================
#         TP:33.83%
#         TN:43.78%
#         FP:5.97%
#         FN:15.92%
#         准确率 ACC:78.00%
#         精确率 PPV:85.00%
#         灵敏度 TPR:68.00%
#         特异度 TNR:88.00%
# =================================
# Top1 - Top10: [77.78, 81.48, 82.72, 83.95, 83.95, 83.95, 83.95, 83.95, 83.95, 83.95]
