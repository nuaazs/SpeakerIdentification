import time
import os
import urllib.request as request
import pandas as pd
import numpy as np
import shutil
import logging
from tqdm import tqdm
from pymysql import *


def get_phone_number(uuid,conn):
    cs1 = conn.cursor()
    count = cs1.execute(f"select caller_num from cti_cdr_call where call_uuid='{uuid}' limit 1")
    if count==0:
        cs1.close()
        return None
    for i in range(count):
        result = cs1.fetchone()
        
        phone_number = result[0]
        #print(phone_number)
    if len(phone_number)<10:
        cs1.close()
        return None
        
    cs1.close()
    return phone_number



info = []
conn = connect(host='116.62.120.233',port=3306,user='changjiangsd',password='changjiangsd9987',database='cticdr',charset='utf8')


df=pd.read_csv("./cti_record.csv")
array = df.to_numpy()
print("读取成功！")
save_root_path = "/mnt/cti_record_data_with_phone_num/"
os.makedirs(save_root_path,exist_ok=True)
def get_timestemp(time_str):
    _time = time.strptime(time_str,"%d/%m/%Y %H:%M:%S")
    return time.mktime(_time)
    
prefix = "http://116.62.120.233/mpccApi/common/downloadFile.json?type=0&addr="
download_info = []
array_rev = array[::-1]

for item in tqdm(array_rev):
    try:
        time_span = get_timestemp(item[3])-get_timestemp(item[2])
    except:
        time_span=0
    if time_span<30:
        continue
    filename = item[8]
    customer_uuid = item[7]
    phone = get_phone_number(customer_uuid,conn)
    customer_uuid = customer_uuid.replace("@192.168.2.230","")
    if phone:
        url = prefix+filename
        filename_short = filename.split("/")[-1]
        father_path = os.path.join(save_root_path,phone,customer_uuid)
        save_path = os.path.join(father_path,filename_short)
        info.append([url,save_path])
        # print([url,save_path])
        #request.urlretrieve(url, filename=save_path)
info = np.array(info)
np.save("info.npy",info)

conn.close()
