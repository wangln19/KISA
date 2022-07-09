import os
import pandas as pd
import random

basic_path = "E:\\数据集\\蚂蚁数据\data\\split_data\\antfin\\HK_sample\\HK_sample\\csv"
delete_ratio = 0.2

for curr_path in os.listdir(basic_path):
    if "train" in curr_path:
        print(curr_path)
        df = pd.read_csv(os.path.join(basic_path,curr_path), encoding='utf-8', sep=',', engine='python', error_bad_lines=False)
        print("before shape:", df.shape)
        label_0 = []
        label_1 = []
        for target_event_id, tmp_df in df.groupby("target_event_id"):
            if tmp_df["label"].iloc[0] == 0:
                label_0.append(target_event_id)
            elif tmp_df["label"].iloc[0] == 1:
                label_1.append(target_event_id)
            else:
                raise ValueError("label incorrect.")


        label_0_delete_list = random.sample(label_0,int(len(label_0)*delete_ratio))
        label_1_delete_list = random.sample(label_1,int(len(label_1)*delete_ratio))

        delete_list = []
        for target_event_id in label_0_delete_list:
            delete_list += list(df[df["target_event_id"]==target_event_id].index)

        for target_event_id in label_1_delete_list:
            delete_list += list(df[df["target_event_id"]==target_event_id].index)

        df = df.drop(delete_list)
        print("after shape:", df.shape)   
        df.to_csv(os.path.join(basic_path,curr_path),index=False)
