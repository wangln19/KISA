"""
train lstm model
record model, embedding and label
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter, writer
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import argparse
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, model_name='checkpoints.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_name = model_name

    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 0,
                 "best_auc": 1, "best_spauc": 1, "latest_update_epoch": 0}
        # save the model with min val loss, so indexes like epoch, beat_auc... is meaningless
        torch.save(state, self.model_name.replace('.pt', '_selected_by_val_loss.pt'))
        self.val_loss_min = val_loss


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


def collate_fn(batch):
    inputs, labels, lengths = zip(*batch)
    inputs_pad = pad_sequence([torch.from_numpy(x) for x in inputs], padding_value=0)
    return inputs_pad.float().to(device), torch.LongTensor(labels).to(device), torch.LongTensor(lengths).to(device)


class LSTMTransfer(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num):
        super(LSTMTransfer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num)
        self.fc = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.8)
        self.src_weight = nn.Parameter(torch.randn(hidden_size, 128))
        self.tgt_weight = nn.Parameter(torch.randn(hidden_size, 128))

        # self.src_weight = nn.Linear(hidden_size, 128, bias=True)
        # self.tgt_weight = nn.Linear(hidden_size, 128, bias=True)

    def forward(self, x, x_length):
        x = pack_padded_sequence(x, x_length.cpu(), enforce_sorted=False)
        x, hidden = self.lstm(x)
        output, _ = pad_packed_sequence(x)
        seq_len, batch_size, hidden_size = output.shape
        output = output.contiguous().view(batch_size * seq_len, hidden_size)
        output = output.view(seq_len, batch_size, -1)

        representation = []
        for i, length in enumerate(x_length):
            representation.append(output[length - 1, i, :])
        representation = torch.stack(representation, dim=0)

        return representation, self.dropout(self.fc(representation))


class TransferLoss(nn.Module):
    def __init__(self, src_hour_list, src_hour_centroid_list, gamma=0.001):
        super(TransferLoss, self).__init__()
        self.src_num_spaces = len(src_hour_list)
        self.rep_hidden_states = len(src_hour_centroid_list[0])
        self.src_hour_centroid_list = [torch.from_numpy(item).cuda(device).reshape((1, -1)) for item in
                                        src_hour_centroid_list]
        self.src_hour_centroid_matrix = torch.cat(self.src_hour_centroid_list, axis=0)  # (src_num_spaces, hidden_states)
        # print("self.src_hour_centroid_matrix:", self.src_hour_centroid_matrix.shape)
        self.gamma = gamma  # trade-off parameters
        # print("gamma", self.gamma)
        self.cls = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.8]).to(device))

    def calc_representation_distance(self):
        distance = torch.zeros((self.src_num_spaces, self.tgt_num_space)).cuda(device)
        for i in range(self.src_num_spaces):
            for j in range(i, self.tgt_num_space):
                distance[i, j] = F.pairwise_distance(self.src_hour_centroid_matrix[i:i + 1],
                                                     self.tgt_hour_centroid_matrix[j:j + 1], p=2)
                distance[j, i] = distance[i, j]
        return distance

    def update_tgt_representation(self, tgt_hour_list, tgt_hour_centroid_list, model):
        self.tgt_num_space = len(tgt_hour_list)
        self.tgt_hour_centroid_matrix = torch.stack(tgt_hour_centroid_list, axis=0)  # (tgt_num_spaces, hidden_states)
        # print("self.src_hour_centroid_matrix:", self.src_hour_centroid_matrix)
        # print("self.tgt_hour_centroid_matrix:", self.tgt_hour_centroid_matrix)

        src_after_trans = torch.mm(self.src_hour_centroid_matrix,
                                   model.src_weight)  # centroid representation in src domain
        tgt_after_trans = torch.mm(self.tgt_hour_centroid_matrix,
                                   model.tgt_weight)  # centroid representation in tgt domain

        # src_after_trans = model.src_weight(self.src_hour_centroid_matrix)
        # tgt_after_trans = model.tgt_weight(self.tgt_hour_centroid_matrix)

        dot_product = torch.mm(src_after_trans, tgt_after_trans.T) / np.sqrt(self.rep_hidden_states)
        alpha = F.softmax(dot_product, dim=0)
        distance = self.calc_representation_distance()

        # print("alpha:", alpha)
        # print("alpha:", alpha.device)
        # print("distance:", distance.device)

        match_loss = torch.mul(alpha, distance).sum().sum()
        # print("match_loss:", match_loss)
        self.match_loss = match_loss

    def forward(self, prob, labels, tgt_hour_list, tgt_hour_centroid_list, model):
        cls_loss = self.cls(prob, labels)
        self.update_tgt_representation(tgt_hour_list, tgt_hour_centroid_list, model)
        return cls_loss + self.gamma * self.match_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.crossEntropy = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        BCE_loss = self.crossEntropy(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def train(model, train_loader, optimizer, epoch, tgt_hour_list, tgt_hour_centroid_list, loss_func=nn.CrossEntropyLoss, desc='Train'):
    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    model.train()
    with tqdm(enumerate(train_loader), desc=desc) as loop:
        for i, batch in loop:
            inputs, labels, lengths = batch
            model.zero_grad()
            rep, prob = model(inputs, lengths)
            logits = torch.argmax(prob, dim=-1)
            loss = loss_func(prob, labels, tgt_hour_list, tgt_hour_centroid_list, model)
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            batch_accuracy = (logits == labels).float().sum().item()
            train_accuracy += batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

    loop.set_postfix(epoch=epoch, loss=train_loss / train_epoch_size, acc=train_accuracy / train_epoch_size)
    writer.add_scalar('loss/train_loss', np.mean(train_loss), epoch)


def eval(model, eval_loader, optimizer, epoch, tgt_hour_list, tgt_hour_centroid_list, loss_func=nn.CrossEntropyLoss, desc='Validation', verbose=True,
         model_name='best_model.pt'):
    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0
    label_list = []
    prob_list = []
    logit_list = []

    model.eval()

    with torch.no_grad():
        with tqdm(enumerate(eval_loader), desc=desc) as loop:
            for i, batch in loop:
                inputs, labels, lengths = batch
                rep, output = model(inputs, lengths)
                logits = torch.argmax(output, dim=-1)

                loss = loss_func(output, labels, tgt_hour_list, tgt_hour_centroid_list, model)

                label_list.append(labels.cpu().numpy())
                prob_list.append(torch.softmax(output, dim=-1)[..., 1].cpu().numpy())
                logit_list.append(logits.cpu().numpy())

                batch_accuracy = (logits == labels).float().sum().item()
                validation_accuracy += batch_accuracy
                validation_epoch_size += 1
                validation_loss += loss.item() * batch_size
                loop.set_postfix(epoch=epoch, loss=validation_loss / validation_epoch_size,
                                 acc=validation_accuracy / validation_epoch_size)

    writer.add_scalar('loss/val_loss', np.mean(validation_loss), epoch)

    global best_auc
    global latest_update_epoch
    # print("label_list:",type(label_list[-1][0]),label_list[-1])
    # print("prob_list:",type(prob_list[-1][0]),prob_list[-1])
    auc = roc_auc_score(label_list, prob_list)
    spauc = roc_auc_score(label_list, prob_list, max_fpr=0.01)
    print(f'Epoch {epoch}, Validation AUC: {auc}, Validation SPAUC: {spauc}')
    '''more details:
    if verbose:
        print(f'Epoch {epoch}, Validation AUC: {auc}, Validation SPAUC: {spauc}')
        print(classification_report(label_list, logit_list, target_names=['0', '1']))'''

    if auc > best_auc:
        best_auc = auc
        latest_update_epoch = epoch
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, "best_auc": auc,
                 "best_spauc": spauc, "latest_update_epoch": latest_update_epoch}
        torch.save(state, model_name)
        print("Updating model... best auc is {}, best spauc is:{}".format(auc, spauc))
        # saving prediction results
        with open(model_name + ".prediction.dict", "wb") as fp:
            pickle.dump({"prob_list": prob_list, "logit_list": logit_list}, fp)
    else:
        # update epoch
        checkpoint = torch.load(model_name)
        state = {'model': checkpoint["model"], 'optimizer': checkpoint["optimizer"], 'epoch': epoch,
                 "best_auc": checkpoint["best_auc"], "best_spauc": checkpoint["best_spauc"],
                 "latest_update_epoch": latest_update_epoch}
        torch.save(state, model_name)
        print("Updating epoch... auc is {}, best spauc is:{}".format(checkpoint["best_auc"], checkpoint["best_spauc"]))
    return validation_loss


def eval_wo_update(model, loader, desc='Validation'):
    validation_accuracy = 0
    validation_epoch_size = 0
    label_list = []
    prob_list = []
    logit_list = []
    rep_list = []

    model.eval()

    with torch.no_grad():
        with tqdm(enumerate(loader), desc=desc) as loop:
            for i, batch in loop:
                inputs, labels, lengths = batch
                rep, output = model(inputs, lengths)
                # print("rep",type(rep),rep.shape)
                logits = torch.argmax(output, dim=-1)

                label_list.append(labels.cpu().numpy())
                prob_list.append(torch.softmax(output, dim=-1)[..., 1].cpu().numpy())
                logit_list.append(logits.cpu().numpy())
                rep_list.append(rep)

                batch_accuracy = (logits == labels).float().sum().item()
                validation_accuracy += batch_accuracy
                validation_epoch_size += 1

                loop.set_postfix(acc=validation_accuracy / validation_epoch_size)

    auc = roc_auc_score(label_list, prob_list)
    spauc = roc_auc_score(label_list, prob_list, max_fpr=0.01)
    precision, recall, thresholds = precision_recall_curve(label_list, prob_list)
    sns.set()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    print(f'min Threshold: {thresholds[0]}, max Threshold: {thresholds[-1]}')
    print(f'Validation AUC: {auc}, Validation SPAUC: {spauc}')
    print(classification_report(label_list, logit_list, target_names=['0', '1']))


def generate_cluster_label(datasets, datafile, num_of_clusters=6):
    with open('E:\Transfer_Learning\Transfer_github\Transfer/retrieve_time_running_record\{}_time.pkl'.format(datasets), "rb") as fp:
        time = pickle.load(fp)
    df = pd.read_csv(datafile)
    ts = []
    dataset = []
    df_group = df.groupby(['target_event_id'], sort=False)
    for target_event_id, frame in df_group:
        if frame['rn'].iloc[0] != 1:
            continue
        ts.append(pd.to_datetime(time[target_event_id]))
    for _ in range(len(ts)):
        ts[_] = (3600*ts[_].hour + 60*ts[_].minute + ts[_].second)*2*math.pi/3600/24
        dataset.append([math.cos(ts[_]), math.sin(ts[_])])
    dataset = np.array(dataset)
    estimator = KMeans(num_of_clusters)  # 构造聚类器
    estimator.fit(dataset)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    centroids = estimator.cluster_centers_  # 获取聚类中心
    inertia = estimator.inertia_
    return label_pred, centroids, inertia


def generate_tgt_representation(model, dataset, tgt_label_list):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    label_list = []
    rep_list = []
    model.eval()
    with torch.no_grad():
        with tqdm(enumerate(loader), desc="loading representation...") as loop:
            for i, batch in loop:
                inputs, labels, lengths = batch
                rep, _ = model(inputs, lengths)
                rep_list.append(rep)
                label_list.append(labels.cpu().numpy())

    reps = torch.stack(rep_list, axis=0).squeeze()
    # labels = np.array(label_list)

    hour_indicator = np.array(tgt_label_list)

    hour_list = np.unique(hour_indicator)
    hour_centroid_list = []
    for lab in hour_list:
        indices = np.where(hour_indicator == lab)[0]
        hour_centroid_list.append(reps[indices].mean(axis=0))
    return hour_list, hour_centroid_list


def load_source_domain_representation(src_label_list, src_model_name):
    """
    return src_rep
    """
    with open(src_model_name.replace(".pt", "_rep.pkl"), "rb") as fp:
        data = pickle.load(fp)
        labels = np.array(data["label"])
        reps = data["rep"]
    hour_indicator = np.array(src_label_list)

    hour_list = np.unique(hour_indicator)
    hour_centroid_list = []
    for lab in hour_list:
        indices = np.where(hour_indicator == lab)[0]
        hour_centroid_list.append(reps[indices].mean(axis=0))

    return hour_list, hour_centroid_list


input_size = None
hidden_size = 300
layer_num = 2
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_auc = 0
latest_update_epoch = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')  # validate
    parser.add_argument('--src_root', default='E:\Transfer_Learning\Data\LZD\csv')
    parser.add_argument('--tgt_root', default='E:\Transfer_Learning\Data\HK\csv')
    parser.add_argument('--filepath', default='train_2020-01.csv')
    parser.add_argument('--lr', default=0.00001, type=float)  # 0.001 for LZD
    parser.add_argument('--src_datasets', default='LZD')
    parser.add_argument('--tgt_datasets', default='HK')
    parser.add_argument('--maxepoch', default=100, type=int)  # 200 for LZD
    parser.add_argument('--loss', default='cross_entropy')
    parser.add_argument('--mark', default="sub_by_hour", type=str)
    parser.add_argument('--num_of_clusters', default=6, type=int)
    parser.add_argument('--gamma', default=0.001, type=float)
    args = parser.parse_args()

    src_trainpath = os.path.join(args.src_root, args.filepath)
    src_evalpath = os.path.join(args.src_root, args.filepath.replace("train", "val"))
    src_testpath = os.path.join(args.src_root, args.filepath.replace("train", "test"))
    src_trainOutputName = args.src_datasets + "_" + os.path.basename(src_trainpath).replace(".csv",
                                                                                            "") + "(finetune).pkl"
    src_evalOutputName = args.src_datasets + "_" + os.path.basename(src_evalpath).replace(".csv", "") + "(finetune).pkl"
    src_testOutputName = args.src_datasets + "_" + os.path.basename(src_testpath).replace(".csv", "") + "(finetune).pkl"
    src_model_name = args.src_datasets + "_" + src_testpath.split("/")[-1].split("_")[-1].replace(".csv",
                                                                                                  "") + "_finetune" + ".pt"
    tgt_trainpath = os.path.join(args.tgt_root, args.filepath)
    tgt_evalpath = os.path.join(args.tgt_root, args.filepath.replace("train", "val"))
    tgt_testpath = os.path.join(args.tgt_root, args.filepath.replace("train", "test"))
    tgt_trainOutputName = args.tgt_datasets + "_" + os.path.basename(tgt_trainpath).replace(".csv",
                                                                                            "") + "(finetune).pkl"
    tgt_evalOutputName = args.tgt_datasets + "_" + os.path.basename(tgt_evalpath).replace(".csv", "") + "(finetune).pkl"
    tgt_testOutputName = args.tgt_datasets + "_" + os.path.basename(tgt_testpath).replace(".csv", "") + "(finetune).pkl"
    tgt_model_name = args.tgt_datasets + "_" + tgt_testpath.split("/")[-1].split("_")[-1].replace(".csv",
                                                                                                  "") + "_" + args.mark + ".pt"
    # writer = SummaryWriter(model_name.replace("pt",""))
    print("---------------------------------------------------")
    print("src train output name:", src_trainOutputName)
    print("src val output name:", src_evalOutputName)
    print("src test output name:", src_testOutputName)
    print("src model name:", src_model_name)
    print("tgt train output name:", tgt_trainOutputName)
    print("tgt val output name:", tgt_evalOutputName)
    print("tgt test output name:", tgt_testOutputName)
    print("tgt model name:", tgt_model_name)
    print("device", device)
    print("---------------------------------------------------")

    writer = SummaryWriter(tgt_model_name.replace("pt", ""))
    # loading tgt train set
    if os.path.exists(tgt_trainOutputName):
        with open(tgt_trainOutputName, "rb") as fp:
            train_dataset = pickle.load(fp)
    else:
        train_dataset = EncodedDataset(tgt_trainpath, src_trainpath)
        with open(tgt_trainOutputName, "wb") as fp:
            pickle.dump(train_dataset, fp)
    # loading tgt val set
    if os.path.exists(tgt_evalOutputName):
        with open(tgt_evalOutputName, "rb") as fp:
            val_dataset = pickle.load(fp)
    else:
        val_dataset = EncodedDataset(tgt_evalpath, src_evalpath)
        with open(tgt_evalOutputName, "wb") as fp:
            pickle.dump(val_dataset, fp)
    # loading test set
    if os.path.exists(tgt_testOutputName):
        with open(tgt_testOutputName, "rb") as fp:
            test_dataset = pickle.load(fp)
    else:
        test_dataset = EncodedDataset(tgt_testpath, src_testpath)
        with open(tgt_testOutputName, "wb") as fp:
            pickle.dump(test_dataset, fp)

    input_size = train_dataset[0][0].shape[1]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = LSTMTransfer(input_size, hidden_size, layer_num).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=float(args.lr), lr_decay=0, weight_decay=0,
                              initial_accumulator_value=0)
    start = 0
    patience = 20

    if os.path.exists(src_model_name):
        src_checkpoint = torch.load(src_model_name)
        src_model = src_checkpoint['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in src_model.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('exist {}!'.format(src_model_name))
    else:
        raise FileNotFoundError("initial source model not found.")

    if os.path.exists(tgt_model_name):
        checkpoint = torch.load(tgt_model_name)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
        best_auc = checkpoint["best_auc"]
        latest_update_epoch = checkpoint["latest_update_epoch"]
        print('exist {}! restart from {}'.format(tgt_model_name, start))

    src_label_list, src_centroids, inertia = generate_cluster_label(args.src_datasets, src_trainpath, args.num_of_clusters)
    src_hour_list, src_hour_centroid_list = load_source_domain_representation(src_label_list, src_model_name)
    tgt_label_list, tgt_centroids, inertia = generate_cluster_label(args.tgt_datasets, tgt_trainpath, args.num_of_clusters)
    tgt_hour_list, tgt_hour_centroid_list = generate_tgt_representation(model, train_dataset, tgt_label_list)
    loss_func = TransferLoss(src_hour_list, src_hour_centroid_list, gamma=float(args.gamma))
    loss_func.update_tgt_representation(tgt_hour_list, tgt_hour_centroid_list, model)

    if args.mode == 'train':
        for epoch in range(start, int(args.maxepoch)):
            train(model, train_loader, optimizer, epoch, tgt_hour_list, tgt_hour_centroid_list, loss_func=loss_func)
            val_loss = eval(model, eval_loader, optimizer, epoch, tgt_hour_list, tgt_hour_centroid_list,
                            loss_func=loss_func, model_name=tgt_model_name)
            tgt_hour_list, tgt_hour_centroid_list = generate_tgt_representation(model, train_dataset, tgt_label_list)
            loss_func.update_tgt_representation(tgt_hour_list, tgt_hour_centroid_list, model)

            early_stopping = EarlyStopping(patience, verbose=False, model_name=tgt_model_name)
            early_stopping(val_loss, model, optimizer)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        checkpoint = torch.load(tgt_model_name)
        model.load_state_dict(checkpoint["model"])
        print(f'latest update epoch: {latest_update_epoch}')
        print("evaluating val set...")
        eval_wo_update(model, eval_loader)
        print("evaluating test set...")
        eval_wo_update(model, test_loader)

    elif args.mode == 'validate':
        model.load_state_dict(checkpoint["model"])
        print("evaluating val set...")
        eval_wo_update(model, eval_loader)
        print("evaluating test set...")
        eval_wo_update(model, test_loader)

