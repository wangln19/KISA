
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import datetime

parseDate = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

with open("LZD_time.pkl","rb") as fp:
    timeMap = pickle.load(fp)

'''prct_ls = [5, 10, 20, 40, 60, 80]
for prct in prct_ls:
    datafile = "E:\\chenliyue\\ant\\sliding_data\\HK_sample_{}\\csv\\train_2020-02.csv".format(prct)
    print("datafile:",datafile)
    df = pd.read_csv(datafile, encoding='utf-8', sep=',', engine='python', error_bad_lines=False).drop(['apdid', 'routermac'], axis=1)
    #target_event_id_list = []
    time_list = []
    df_group = df.groupby(['target_event_id'], sort=False)
    drop_features = ['rn', 'target_event_id', 'label']
    input_size = df.shape[1] - len(drop_features)
    with tqdm(df_group, desc='loading data...') as loop:
        for target_event_id, frame in loop:
            if frame['rn'].iloc[0] != 1:
                continue
            #target_event_id_list.append(frame['target_event_id'].iloc[0])
            time_list.append(timeMap[target_event_id])

    time_list = [parseDate(x) for x in time_list]
    month_list = [x.month for x in time_list]
    month_list = np.array(month_list)

    with open("HK_2020-01_month_{}.pkl".format(prct),"wb") as fp:
        pickle.dump(month_list,fp)'''


datafile = "E:\Transfer_Learning\Data\LZD\csv/train_2020-01.csv"
print("datafile:",datafile)
df = pd.read_csv(datafile, encoding='utf-8', sep=',', engine='python', error_bad_lines=False).drop(['apdid', 'routermac'], axis=1)
#target_event_id_list = []
time_list = []
df_group = df.groupby(['target_event_id'], sort=False)
drop_features = ['rn', 'target_event_id', 'label']
input_size = df.shape[1] - len(drop_features)
with tqdm(df_group, desc='loading data...') as loop:
    for target_event_id, frame in loop:
        if frame['rn'].iloc[0] != 1:
            continue
        #target_event_id_list.append(frame['target_event_id'].iloc[0])
        time_list.append(timeMap[target_event_id])

time_list = [parseDate(x) for x in time_list]
month_list = [x.month for x in time_list]
month_list = np.array(month_list)
with open("LZD_2020-02_month.pkl","wb") as fp:
    pickle.dump(month_list,fp)