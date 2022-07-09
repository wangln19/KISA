
import os
import pandas as pd
from tqdm import tqdm
import pickle

datafile = "E:\\数据集\\蚂蚁数据\\data\\HK迁移学习数据输出02.csv"

df = pd.read_csv(datafile, encoding='utf-8', sep=',', engine='python', error_bad_lines=False)

event2time = {}

for idx,tmp in tqdm(df.groupby("target_event_id")):
    event2time[idx] = tmp["target_gmt_occur_cn"].iloc[0]

with open(os.path.basename(datafile).replace("迁移学习数据输出02.csv","") + "_time.pkl","wb") as fp:
    pickle.dump(event2time,fp)