import time
import os
import urllib.request as request
import pandas as pd
import numpy as np
import shutil
import logging
from tqdm import tqdm
from pymysql import *

import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description='')
parser.add_argument('--index', type=int, default="0",help='')
args = parser.parse_args()
start = (args.index)*10000
end = (args.index+1)*10000

info = np.load("info.npy")



logging.basicConfig(filename=f"./download_index_{args.index}.log", level=logging.DEBUG)

save_root_path = "/mnt/cti_record_data_with_phone_num/"
os.makedirs(save_root_path,exist_ok=True)

print(len(info))
print(f"Downloading : {start} ~ {end}")
for item in tqdm(info[start:end]):
    url,save_path = item
    path = Path(save_path)
    if os.path.exists(path):
        continue
    father = path.parent
    os.makedirs(father,exist_ok=True)
    # print(father)
    try:
        request.urlretrieve(url, filename=save_path)
    except Exception as e:
        logging.error(f"Error: {url} -> message:{e}")

    logging.info(f"\t保存成功： {url} -> {save_path}")