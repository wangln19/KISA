"""
Dataset Classes
"""

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import re


class EncodedDataset(Dataset):
    def __init__(self, datafile, cmp_datafile):
        super(EncodedDataset, self).__init__()
        self.data = []
        self.label = []
        self.length = []

        global input_size
        df = pd.read_csv(datafile, encoding='utf-8', sep=',', engine='python', error_bad_lines=False).drop(
            ['apdid', 'routermac'], axis=1)
        cmp_df = pd.read_csv(cmp_datafile, encoding='utf-8', sep=',', engine='python', error_bad_lines=False).drop(
            ['apdid', 'routermac'], axis=1)
        # Normalization
        arr = np.array(df['event_amount'])
        lef = np.mean(arr) - 3 * np.std(arr)
        rgt = np.mean(arr) + 3 * np.std(arr)
        lef = min(arr) if min(arr) > lef else lef
        rgt = max(arr) if max(arr) < rgt else rgt
        df['event_amount'] = df['event_amount'].apply(lambda x: rgt if x > rgt else x)
        df['event_amount'] = df['event_amount'].apply(lambda x: lef if x < lef else x)
        df['event_amount'] = df['event_amount'].apply(lambda x: (x - lef) / (rgt - lef))

        df_col = list(df.columns)
        cmp_df_col = list(cmp_df.columns)
        drop_features = [_ for _ in df_col if _ not in cmp_df_col]
        drop_features += ['rn', 'target_event_id', 'label']
        df_group = df.groupby(['target_event_id'], sort=False)
        input_size = df.shape[1] - len(drop_features)
        print('input size ', input_size)
        with tqdm(df_group, desc='loading data...') as loop:
            for target_event_id, frame in loop:
                if frame['rn'].iloc[0] != 1:
                    continue
                self.label.append(frame['label'].iloc[0])
                frame.sort_values(['rn'], inplace=True, ascending=False)
                x = frame.drop(drop_features, axis=1).to_numpy()
                self.data.append(x)
                self.length.append(len(x))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item], self.length[item]


class FusionDataset(Dataset):
    def __init__(self, rep_name1, rep_name2):
        super(FusionDataset, self).__init__()
        self.rep1 = []
        self.rep2 = []
        self.label = []
        with open(rep_name1, "rb") as fp:
            data = pickle.load(fp)
            self.rep1 = data["rep"]
            self.label = np.array(data["label"])
            
        with open(rep_name2, "rb") as fp:
            data = pickle.load(fp)
            self.rep2 = data["rep"]
            
    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.rep1[item], self.rep2[item], self.label[item]


class EncodedDataset_lstm(Dataset):
    def __init__(self, datafile):
        super(EncodedDataset_lstm, self).__init__()
        self.data = []
        self.label = []
        self.length = []

        global input_size
        df = pd.read_csv(datafile, encoding='utf-8', sep=',', engine='python', error_bad_lines=False).drop(
            ['apdid', 'routermac'], axis=1)
        # Normalization
        arr = np.array(df['event_amount'])
        lef = np.mean(arr) - 3 * np.std(arr)
        rgt = np.mean(arr) + 3 * np.std(arr)
        lef = min(arr) if min(arr) > lef else lef
        rgt = max(arr) if max(arr) < rgt else rgt
        df['event_amount'] = df['event_amount'].apply(lambda x: rgt if x > rgt else x)
        df['event_amount'] = df['event_amount'].apply(lambda x: lef if x < lef else x)
        df['event_amount'] = df['event_amount'].apply(lambda x: (x - lef) / (rgt - lef))

        df_group = df.groupby(['target_event_id'], sort=False)
        drop_features = ['rn', 'target_event_id', 'label']
        input_size = df.shape[1] - len(drop_features)
        with tqdm(df_group, desc='loading data...') as loop:
            for target_event_id, frame in loop:
                if frame['rn'].iloc[0] != 1:
                    continue
                self.label.append(frame['label'].iloc[0])
                frame.sort_values(['rn'], inplace=True, ascending=False)
                x = frame.drop(drop_features, axis=1).to_numpy()

                self.data.append(x)
                self.length.append(len(x))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item], self.length[item]


class ShortCut_EncodedDataset(Dataset):
    def __init__(self, datafile, cmp_datafile):
        super(ShortCut_EncodedDataset, self).__init__()
        self.data = []
        self.label = []
        self.length = []
        self.shortcut_var = []

        global input_size
        df = pd.read_csv(datafile, encoding='utf-8', sep=',', engine='python', error_bad_lines=False).drop(
            ['apdid', 'routermac'], axis=1)
        cmp_df = pd.read_csv(cmp_datafile, encoding='utf-8', sep=',', engine='python', error_bad_lines=False).drop(
            ['apdid', 'routermac'], axis=1)
        # Normalization
        arr = np.array(df['event_amount'])
        lef = np.mean(arr) - 3 * np.std(arr)
        rgt = np.mean(arr) + 3 * np.std(arr)
        lef = min(arr) if min(arr) > lef else lef
        rgt = max(arr) if max(arr) < rgt else rgt
        df['event_amount'] = df['event_amount'].apply(lambda x: rgt if x > rgt else x)
        df['event_amount'] = df['event_amount'].apply(lambda x: lef if x < lef else x)
        df['event_amount'] = df['event_amount'].apply(lambda x: (x - lef) / (rgt - lef))

        df_col = list(df.columns)
        cmp_df_col = list(cmp_df.columns)
        drop_features = [_ for _ in df_col if _ not in cmp_df_col]
        drop_features += ['rn', 'target_event_id', 'label']
        df_group = df.groupby(['target_event_id'], sort=False)
        input_size = df.shape[1] - len(drop_features)
        # print('input size ', input_size) 96

        shortcut_var_list = []
        for _ in list(df.columns):
            if re.match('card_type', _):
                shortcut_var_list.append(_)
            if re.match('hour', _):
                shortcut_var_list.append(_)

        with tqdm(df_group, desc='loading data...') as loop:
            for target_event_id, frame in loop:
                tmp = []
                if frame['rn'].iloc[0] != 1:
                    continue
                self.label.append(frame['label'].iloc[0])
                frame.sort_values(['rn'], inplace=True, ascending=False)
                x = frame.drop(drop_features, axis=1).to_numpy()
                self.data.append(x)
                self.length.append(len(x))
                # print('shortcut_var_list', shortcut_var_list)
                for _ in shortcut_var_list:
                    tmp.append(frame[_].iloc[-1])
                self.shortcut_var.append(np.array(tmp))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item], self.length[item], self.shortcut_var[item]
