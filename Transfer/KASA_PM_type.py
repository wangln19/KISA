import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter, writer
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
from Earlystopping import EarlyStopping
from Loss import coral_loss
from Dataset import EncodedDataset
from Model import LSTMClassifier
import re


class TransferMeanLoss(nn.Module):
    def __init__(self, gamma=0.5, da_gamma=1):
        super(TransferMeanLoss, self).__init__()
        self.gamma = gamma  # trade-off parameters
        self.da_gamma = da_gamma
        # print("gamma", self.gamma)
        self.cls = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.8]).to(device))

    def update_src_representation(self, src_hour_list, src_hour_rep_list):  # src_rep_0, src_rep_1
        self.src_num_space = len(src_hour_list)
        self.rep_hidden_states = src_hour_rep_list[0][0].shape[1]
        self.src_hour_rep_list = src_hour_rep_list
        # self.src_rep_0 = src_rep_0
        # self.src_rep_1 = src_rep_1

    def update_tgt_representation(self, tgt_hour_list, tgt_hour_rep_list):  # tgt_rep_0, tgt_rep_1
        self.tgt_num_space = len(tgt_hour_list)
        self.tgt_hour_rep_list = tgt_hour_rep_list
        # self.tgt_rep_0 = tgt_rep_0
        # self.tgt_rep_1 = tgt_rep_1

    def calc_representation_distance(self):
        dists = []
        for _ in range(self.tgt_num_space):
            dist = []
            for __ in range(self.src_num_space):
                if isinstance(self.src_hour_rep_list[__][0], np.ndarray):
                    src_hour_rep_centroid0 = torch.from_numpy(self.src_hour_rep_list[__][0]).cuda()
                else:
                    src_hour_rep_centroid0 = self.src_hour_rep_list[__][0]
                if isinstance(self.src_hour_rep_list[__][1], np.ndarray):
                    src_hour_rep_centroid1 = torch.from_numpy(self.src_hour_rep_list[__][1]).cuda()
                else:
                    src_hour_rep_centroid1 = self.src_hour_rep_list[__][1]
                if isinstance(self.tgt_hour_rep_list[_][0], np.ndarray):
                    tgt_hour_rep_centroid0 = torch.from_numpy(self.tgt_hour_rep_list[_][0]).cuda()
                else:
                    tgt_hour_rep_centroid0 = self.tgt_hour_rep_list[_][0]
                if isinstance(self.tgt_hour_rep_list[_][1], np.ndarray):
                    tgt_hour_rep_centroid1 = torch.from_numpy(self.tgt_hour_rep_list[_][1]).cuda()
                else:
                    tgt_hour_rep_centroid1 = self.tgt_hour_rep_list[_][1]
                dis = coral_loss(src_hour_rep_centroid0, tgt_hour_rep_centroid0) + coral_loss(src_hour_rep_centroid1, tgt_hour_rep_centroid1) 
                dist.append(float(dis))
            dists.append(dist)
        distance_st = np.array(dists)  # (self.tgt_num_space, self.src_num_spaces)
        '''
        distance_tt = np.zeros((self.tgt_num_space, self.tgt_num_space))
        for _ in range(self.tgt_num_space):
            for __ in range(_+1, self.tgt_num_space):
                if isinstance(self.tgt_hour_rep_list[__], np.ndarray):
                    tgt_hour_rep_centroid1 = torch.from_numpy(self.tgt_hour_rep_list[__]).cuda()
                else:
                    tgt_hour_rep_centroid1 = self.tgt_hour_rep_list[__]
                if isinstance(self.tgt_hour_rep_list[_], np.ndarray):
                    tgt_hour_rep_centroid2 = torch.from_numpy(self.tgt_hour_rep_list[_]).cuda()
                else:
                    tgt_hour_rep_centroid2 = self.tgt_hour_rep_list[_]
                distance_tt[_, __] = coral_loss(tgt_hour_rep_centroid1, tgt_hour_rep_centroid2) 

        distance_ss = np.zeros((self.src_num_space, self.src_num_space))
        for _ in range(self.src_num_space):    
            for __ in range(_+1, self.src_num_space):
                if isinstance(self.src_hour_rep_list[__], np.ndarray):
                    src_hour_rep_centroid1 = torch.from_numpy(self.src_hour_rep_list[__]).cuda()
                else:
                    src_hour_rep_centroid1 = self.src_hour_rep_list[__]
                if isinstance(self.src_hour_rep_list[_], np.ndarray):
                    src_hour_rep_centroid2 = torch.from_numpy(self.src_hour_rep_list[_]).cuda()
                else:
                    src_hour_rep_centroid2 = self.src_hour_rep_list[_]
                distance_ss[_, __] = coral_loss(src_hour_rep_centroid1, src_hour_rep_centroid2)
        '''
        return distance_st

    def calculate_weight_matrix(self):
        weight_matrix1 = np.ones((self.tgt_num_space, self.src_num_space))
        for _ in range(self.tgt_num_space):
            for __ in range(self.src_num_space):
                weight_matrix1[_, __] = (len(self.src_hour_rep_list[__][0]) + len(self.src_hour_rep_list[__][1])) * (len(self.tgt_hour_rep_list[_][0]) + len(self.tgt_hour_rep_list[_][1])) / len_of_src /len_of_tgt
        '''
        weight_matrix2 = np.zeros((self.tgt_num_space, self.tgt_num_space))
        weight_matrix3 = np.zeros((self.src_num_space, self.src_num_space))
        for _ in range(self.tgt_num_space):
            for __ in range(_+1, self.tgt_num_space):
                weight_matrix2[_, __] = len(self.tgt_hour_rep_list[__]) * len(self.tgt_hour_rep_list[_]) / len_of_tgt /len_of_tgt
        for _ in range(self.src_num_space):
            for __ in range(_+1, self.src_num_space):
                weight_matrix3[_, __] = len(self.src_hour_rep_list[__]) * len(self.src_hour_rep_list[_]) / len_of_src /len_of_src
        '''
        return weight_matrix1

    def match_representation(self, top_k=1):
        # dist_matrix_st, dist_matrix_tt, dist_matrix_ss= self.calc_representation_distance()
        dist_matrix_st = self.calc_representation_distance()
        numerator_matrix = np.zeros((self.tgt_num_space, self.src_num_space))
        denominator_matrix = np.ones((self.tgt_num_space, self.src_num_space))
        for _ in range(self.tgt_num_space):
            for __ in range(self.src_num_space):
                if _ == __:
                    numerator_matrix[_, __] = 1
                    denominator_matrix[_, __] = 0
        # weight_matrix_st, weight_matrix_tt, weight_matrix_ss = self.calculate_weight_matrix()
        weight_matrix_st = self.calculate_weight_matrix()
        numerator_matrix = np.multiply(numerator_matrix, weight_matrix_st)
        denominator_matrix_st = np.multiply(denominator_matrix, weight_matrix_st)
        numerator = torch.mul(torch.from_numpy(dist_matrix_st), torch.from_numpy(numerator_matrix)).sum().sum()
        denominator_st = torch.mul(torch.from_numpy(dist_matrix_st), torch.from_numpy(denominator_matrix_st)).sum().sum()
        # denominator_tt = torch.mul(torch.from_numpy(dist_matrix_tt), torch.from_numpy(weight_matrix_tt)).sum().sum()
        # denominator_ss = torch.mul(torch.from_numpy(dist_matrix_ss), torch.from_numpy(weight_matrix_ss)).sum().sum()
        # denominator = denominator_st + denominator_tt + denominator_ss
        denominator = denominator_st
        match_loss = np.log(3 * numerator / denominator)
        self.match_loss = match_loss
    
    def domain_distance(self, rep_1, rep_2):
        if isinstance(rep_1, np.ndarray):
            rep_1 = torch.from_numpy(rep_1).cuda()
        if isinstance(rep_2, np.ndarray):
            rep_2 = torch.from_numpy(rep_2).cuda()
        return F.pairwise_distance(rep_1, rep_2, p=2)  # 2-order distance

    def forward(self, prob, labels):
        cls_loss = self.cls(prob, labels)
        # print('cls:', cls_loss)
        # print('match', self.gamma * self.match_loss)
        # return cls_loss + self.gamma * self.match_loss + self.da_gamma * self.da_loss
        return cls_loss + self.gamma * self.match_loss


def collate_fn(batch):
    inputs, labels, lengths = zip(*batch)
    inputs_pad = pad_sequence([torch.from_numpy(x) for x in inputs], padding_value=0)
    return inputs_pad.float().to(device), torch.LongTensor(labels).to(device), torch.LongTensor(lengths).to(device)


def train(model, train_loader, optimizer, epoch, loss_func=nn.CrossEntropyLoss, desc='Train'):
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
            loss = loss_func(prob, labels)
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            batch_accuracy = (logits == labels).float().sum().item()
            train_accuracy += batch_accuracy
            b_size = inputs.shape[1]
            train_epoch_size += b_size
            train_loss += loss.item() * b_size

    loop.set_postfix(epoch=epoch, loss=train_loss / train_epoch_size, acc=train_accuracy / train_epoch_size)
    writer.add_scalar('loss/train_loss', np.mean(train_loss), epoch)


def eval(model, eval_loader, optimizer, epoch, loss_func=nn.CrossEntropyLoss, desc='Validation', verbose=True,
         model_name='best_model.pt'):
    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0
    match_loss = 0
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

                loss = loss_func(output, labels)

                label_list.append(labels.cpu().numpy())
                prob_list.append(torch.softmax(output, dim=-1)[..., 1].cpu().numpy())
                logit_list.append(logits.cpu().numpy())

                batch_accuracy = (logits == labels).float().sum().item()
                validation_accuracy += batch_accuracy
                validation_epoch_size += 1
                validation_loss += loss.item() 
                match_loss += (loss_func.gamma * loss_func.match_loss).item() 
                loop.set_postfix(epoch=epoch, loss=validation_loss / validation_epoch_size,
                                 acc=validation_accuracy / validation_epoch_size)

    writer.add_scalar('loss/val_loss', np.mean(validation_loss), epoch)
    writer.add_scalar('loss/match_loss', np.mean(match_loss), epoch)

    global best_auc
    global latest_update_epoch
    # print("label_list:",type(label_list[-1][0]),label_list[-1])
    # print("prob_list:",type(prob_list[-1][0]),prob_list[-1])
    auc = roc_auc_score(label_list, prob_list)
    spauc = roc_auc_score(label_list, prob_list, max_fpr=0.01)
    print(f'Epoch {epoch}, Validation AUC: {auc}, Validation SPAUC: {spauc}')
    '''
    more details:
    if verbose:
        print(f'Epoch {epoch}, Validation AUC: {auc}, Validation SPAUC: {spauc}')
        print(classification_report(label_list, logit_list, target_names=['0', '1']))
    '''

    '''
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
    '''
    return np.mean(validation_loss)


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
    print(f'AUC: {auc}, SPAUC: {spauc}')
    print(classification_report(label_list, logit_list, target_names=['0', '1']))


def generate_cluster_label(datasets, datafile):
    num_of_card_type = []
    df = pd.read_csv(datafile)
    all_list = []
    for _ in list(df.columns):
        if re.match('card_type', _):
            all_list.append(_)
    df_group = df.groupby(['target_event_id'], sort=False)
    for target_event_id, frame in df_group:
        if frame['rn'].iloc[0] != 1:
            continue
        ctimes = 0
        for _ in all_list:
            ctimes +=  frame[_].iloc[-1]
        num_of_card_type.append(ctimes)
    return num_of_card_type


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
    hour_rep_list = []
    for lab in hour_list:
        array1 = np.where(hour_indicator == lab)[0] 
        array2 = np.where(labels == 0)[0]
        indices1 = np.intersect1d(array1, array2)
        indices2 = np.setdiff1d(array1, array2)
        hour_rep_list.append([reps[indices1], reps[indices2]])

    return hour_list, hour_rep_list


def generate_representation(model, dataset, input_label_list):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    label_list = []
    rep_list = []
    model.eval()
    with torch.no_grad():
        with tqdm(enumerate(loader), desc="loading representation...") as loop:
            for i, batch in loop:
                inputs, labels, lengths = batch
                rep, _ = model(inputs, lengths)
                for _ in rep:
                    rep_list.append(_)
                for _ in labels:
                    label_list.append(_.cpu().numpy())

    reps = torch.stack(rep_list, axis=0).squeeze()
    labels = np.array(label_list)
    hour_indicator = np.array(input_label_list)

    hour_list = np.unique(hour_indicator)
    hour_rep_list = []
    for lab in hour_list:
        array1 = np.where(hour_indicator == lab)[0] 
        array2 = np.where(labels == 0)[0]
        indices1 = np.intersect1d(array1, array2)
        indices2 = np.setdiff1d(array1, array2)
        hour_rep_list.append([reps[indices1], reps[indices2]])
    return hour_list, hour_rep_list


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
    parser.add_argument('--src_root', default='../Data/LZD/csv')
    parser.add_argument('--tgt_root', default='../Data/HK/csv')
    parser.add_argument('--filepath', default='train_2020-01.csv')
    parser.add_argument('--lr', default=0.0001, type=float)  # 0.001 for LZD
    parser.add_argument('--src_datasets', default='LZD')
    parser.add_argument('--tgt_datasets', default='HK')
    parser.add_argument('--maxepoch', default=2000, type=int)  # 200 for LZD
    parser.add_argument('--loss', default='cross_entropy')
    parser.add_argument('--mark', default="KASA_PM_type", type=str)
    parser.add_argument('--gamma', default=0.01, type=float)  # 0.01
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

    writer = SummaryWriter(tgt_model_name.replace(".pt", ""))
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
    # loading src set
    if os.path.exists(src_trainOutputName):
        with open(src_trainOutputName, "rb") as fp:
            src_dataset = pickle.load(fp)
    else:
        src_dataset = EncodedDataset(src_trainpath, tgt_trainpath)
        with open(src_trainOutputName, "wb") as fp:
            pickle.dump(src_dataset, fp)

    input_size = train_dataset[0][0].shape[1]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = LSTMClassifier(input_size, hidden_size, layer_num).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=float(args.lr), lr_decay=0, weight_decay=0,
                              initial_accumulator_value=0)
    start = 0
    patience = 20
    early_stopping = EarlyStopping(patience, verbose=False, model_name=tgt_model_name)
    len_of_src = 0
    len_of_tgt = 0

    '''
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
    '''

    if os.path.exists(tgt_model_name):
        checkpoint = torch.load(tgt_model_name)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
        best_auc = checkpoint["best_auc"]
        latest_update_epoch = checkpoint["latest_update_epoch"]
        print('exist {}! restart from {}'.format(tgt_model_name, start))

    src_label_list = generate_cluster_label(args.src_datasets, src_trainpath)
    len_of_src = len(src_label_list)
    print('len_of_src', len_of_src)
    src_hour_list, src_hour_rep_list = load_source_domain_representation(src_label_list, src_model_name)
    tgt_label_list = generate_cluster_label(args.tgt_datasets, tgt_trainpath)
    len_of_tgt = len(tgt_label_list)
    print('len_of_tgt', len_of_tgt)
    tgt_hour_list, tgt_hour_rep_list = generate_representation(model, train_dataset, tgt_label_list)
    loss_func = TransferMeanLoss(gamma=float(args.gamma))
    loss_func.update_src_representation(src_hour_list, src_hour_rep_list)
    loss_func.update_tgt_representation(tgt_hour_list, tgt_hour_rep_list)

    if args.mode == 'train':
        for epoch in range(start, int(args.maxepoch)):
            loss_func.match_representation()
            train(model, train_loader, optimizer, epoch, loss_func=loss_func)
            val_loss = eval(model, eval_loader, optimizer, epoch, loss_func=loss_func, model_name=tgt_model_name)

            src_hour_list, src_hour_rep_list = generate_representation(model, src_dataset, src_label_list)
            loss_func.update_src_representation(src_hour_list, src_hour_rep_list)
            tgt_hour_list, tgt_hour_rep_list = generate_representation(model, train_dataset, tgt_label_list)
            loss_func.update_tgt_representation(tgt_hour_list, tgt_hour_rep_list)

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


