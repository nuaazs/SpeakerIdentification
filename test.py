# 依赖
import argparse
from hyperpyyaml import load_hyperpyyaml
import os
import torch
import tqdm
import soundfile as sf
import numpy as np

# SpeechBrain依赖
from speechbrain.pretrained import EncoderClassifier

# 命令解析
parser = argparse.ArgumentParser(description='### VoxCeleb-SpeakerRecognition的argument parser')
parser.add_argument('-y', '--yaml', type=str, default='./eval_cosine.yaml')

if __name__ == '__main__':

    # 命令解析
    args = dict(vars(parser.parse_args()))
    yaml_path = args['yaml']

    # yaml解析
    print('# 解析yaml...')
    print(f'### TODO yaml地址为{yaml_path}')
    with open(yaml_path) as f:
        hparams = load_hyperpyyaml(f)
    #print(hparams)

    # 读取trial_pair
    print('# 读取测试配对trial_pair...')
    print('### TODO trial_pair地址为'+str(hparams['trial_pair']))
    with open(hparams['trial_pair']) as f:
        txt_lines = f.readlines()
    trial_pairs = []
    for line in txt_lines:
        #print(line)
        label, wav1_path, wav2_path = line.split(' ')
        wav2_path = wav2_path.replace('\n', '')
        trial = {}
        trial['label'] = int(label)
        trial['spk1'] = wav1_path.split('/')[0]
        trial['spk1_path'] = os.path.join(hparams['data_dir'], hparams['sub_dir'], wav1_path)
        trial['spk2'] = wav2_path.split('/')[0]
        trial['spk2_path'] = os.path.join(hparams['data_dir'], hparams['sub_dir'], wav2_path)

        #print(trial)
        trial_pairs.append(trial)

    
    cuda = torch.device('cuda:'+str(hparams['GPU']))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams['GPU']) 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # 模型
    print('# 载入模型...')
    print('### TODO 模型地址为'+str(hparams['speechbrain_model']))
    print('### TODO 模型超参数地址为'+str(hparams['model_hparams']))
    print('### TODO 选定GPU为cuda:'+str(hparams['GPU']))
    # 读取训练好的模型作为encoder前端
    spkreg = EncoderClassifier.from_hparams(
        source=hparams['speechbrain_model'],
        hparams_file=hparams['model_hparams'],
        run_opts={'device': 'cuda:0'}) # cuda0指的是允许使用的GPU列表(CUDA_VISIBLE_DEVICES)里的第0个GPU
    torch.cuda.empty_cache()

    # cosine作为打分后端
    print('### 使用cosine作为后端打分')
    similarity = torch.nn.CosineSimilarity(dim=-1,eps=1e-6)

    score_samespk = []
    score_diffspk = []

    print('# 开始评估...')
    print('### TODO: 数据集地址为'+str(hparams['data_dir']+hparams['sub_dir']))
    # 测试评估循环
    for step, pair in enumerate(tqdm.tqdm(trial_pairs)):

        # 参数
        label = pair['label']
        speech1_path = pair['spk1_path']
        speech2_path = pair['spk2_path']
        
        # 读取
        with torch.no_grad():
            wav1, sr1 = sf.read(speech1_path)
            wav2, sr2 = sf.read(speech2_path)

            dur_max = hparams['dur_max']
            if len(wav1)/sr1 > dur_max:
                wav1 = wav1[:dur_max*sr1]
            if len(wav2)/sr2 > dur_max:
                wav2 = wav2[:dur_max*sr2]

            wavs1 = torch.tensor(wav1).unsqueeze(0)#.cuda()
            wavs2 = torch.tensor(wav2).unsqueeze(0)#.cuda()
            wav_batch = torch.cat([wavs1, wavs2]).cuda()

            # 提取embedding
            embedding_batch = spkreg.encode_batch(wav_batch)
            embeddings1 = embedding_batch[0].unsqueeze(0)
            embeddings2 = embedding_batch[1].unsqueeze(0)
            #embeddings1 = spkreg.encode_batch(wavs1)
            #embeddings2 = spkreg.encode_batch(wavs2)

            score = similarity(embeddings1, embeddings2).cpu().numpy()[0]

        #print(label, score)

        if label == 1:
            score_samespk.append(score)
        elif label == 0:
            score_diffspk.append(score)

        if step%5000 == 0:
            np.save(os.path.join(hparams['save_path'], 'score_samespk.npy'), score_samespk)
            np.save(os.path.join(hparams['save_path'], 'score_diffspk.npy'), score_diffspk)

    np.save(os.path.join(hparams['save_path'], 'score_samespk.npy'), score_samespk)
    np.save(os.path.join(hparams['save_path'], 'score_diffspk.npy'), score_diffspk)

    print('# done.')
    from IPython import embed
    embed()
    torch.cuda.empty_cache()
