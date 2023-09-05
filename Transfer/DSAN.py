import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter, writer
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
import pandas as pd
import numpy as np
import pickle
import random
import os
from tqdm import tqdm
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
from Earlystopping import EarlyStopping
from Loss import LMMD_loss
from Dataset import EncodedDataset
from Model import LSTMClassifier


class TransferLMMDLoss(nn.Module):
    def __init__(self, gamma=0.001):
        super(TransferLMMDLoss, self).__init__()
        self.gamma = gamma  # trade-off parameters
        # print("gamma", self.gamma)
        self.cls = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.8]).to(device))

    def update_src_representation(self, src_hour_rep):
        self.rep_hidden_states = src_hour_rep.shape[1]
        self.src_hour_rep = src_hour_rep

    def update_label(self, src_label, tgt_label):
        self.src_label = src_label
        self.tgt_label = tgt_label

    def update_tgt_representation(self, tgt_hour_rep):
        self.tgt_hour_rep = tgt_hour_rep

    def select_src_representation(self, num):
        count = 0
        src_hour_rep = self.src_hour_rep
        if isinstance(src_hour_rep, torch.Tensor):
            src_hour_rep = src_hour_rep.cpu().numpy()
        while True:
            indices = random.sample(range(src_hour_rep.shape[0]), num)
            slt_src_hour_rep = src_hour_rep[indices]
            selected_centroid = slt_src_hour_rep.mean(axis=0)
            ori_centroid = src_hour_rep.mean(axis=0)
            dist = F.pairwise_distance(torch.from_numpy(ori_centroid.reshape(1, self.rep_hidden_states)),
                                       torch.from_numpy(selected_centroid.reshape(1, self.rep_hidden_states)), p=2)
            dist = float(dist)
            # set select bar = 0.1
            if dist < 0.1 or num < 30:
                break
            else:
                count += 1
                if count > 50:
                    raise RuntimeError("Select Centroid Bar is too Strict.")
        return slt_src_hour_rep, self.src_label[indices].squeeze()

    def select_tgt_representation(self, num):
        count = 0
        tgt_hour_rep = self.tgt_hour_rep
        if isinstance(tgt_hour_rep, torch.Tensor):
            tgt_hour_rep = tgt_hour_rep.cpu().numpy()
        while True:
            indices = random.sample(range(tgt_hour_rep.shape[0]), num)
            slt_tgt_hour_rep = tgt_hour_rep[indices]
            selected_centroid = slt_tgt_hour_rep.mean(axis=0)
            ori_centroid = tgt_hour_rep.mean(axis=0)
            dist = F.pairwise_distance(torch.from_numpy(ori_centroid.reshape(1, self.rep_hidden_states)),
                                       torch.from_numpy(selected_centroid.reshape(1, self.rep_hidden_states)), p=2)
            dist = float(dist)
            # set select bar = 0.1
            if dist < 0.1 or num < 30:
                break
            else:
                count += 1
                if count > 50:
                    raise RuntimeError("Select Centroid Bar is too Strict.")
        
        return slt_tgt_hour_rep, self.tgt_label[indices].squeeze()

    def calc_representation_distance(self):
        slt_src_rep, slt_src_label = self.select_src_representation(len(self.tgt_hour_rep)//6) # //6 to save the memory
        slt_tgt_rep, slt_tgt_label = self.select_tgt_representation(len(self.tgt_hour_rep)//6)
        if isinstance(slt_src_label, np.ndarray):
            slt_src_label = torch.from_numpy(slt_src_label).cuda()
        if isinstance(slt_src_rep, np.ndarray):
            slt_src_rep = torch.from_numpy(slt_src_rep).cuda()
        if isinstance(slt_tgt_label, np.ndarray):
            slt_tgt_label = torch.from_numpy(slt_tgt_label).cuda()
        if isinstance(slt_tgt_rep, np.ndarray):
            slt_tgt_rep = torch.from_numpy(slt_tgt_rep).cuda()
        lmmd = LMMD_loss()
        distance = lmmd.get_loss(slt_src_rep, slt_tgt_rep, slt_src_label, slt_tgt_label)
        self.match_loss = distance

    def forward(self, prob, labels):
        cls_loss = self.cls(prob, labels)
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

    global best_spauc
    global latest_update_epoch
    # print("label_list:",type(label_list[-1][0]),label_list[-1])
    # print("prob_list:",type(prob_list[-1][0]),prob_list[-1])
    auc = roc_auc_score(label_list, prob_list)
    spauc = roc_auc_score(label_list, prob_list, max_fpr=0.1)
    print(f'Epoch {epoch}, Validation AUC: {auc}, Validation SPAUC: {spauc}')
    '''
    more details:
    if verbose:
        print(f'Epoch {epoch}, Validation AUC: {auc}, Validation SPAUC: {spauc}')
        print(classification_report(label_list, logit_list, target_names=['0', '1']))
    '''

    model_name = model_name.replace('.pt', '_slt_by_spauc.pt')
    if spauc > best_spauc:
        best_spauc = spauc
        latest_update_epoch = epoch
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, "auc": auc,
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
                 "auc": checkpoint["auc"], "best_spauc": checkpoint["best_spauc"],
                 "latest_update_epoch": latest_update_epoch}
        torch.save(state, model_name)
        print("Updating epoch... auc is {}, best spauc is:{}".format(checkpoint["auc"], checkpoint["best_spauc"]))

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
    spauc = roc_auc_score(label_list, prob_list, max_fpr=0.1)
    precision, recall, thresholds = precision_recall_curve(label_list, prob_list)
    sns.set()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    print(f'min Threshold: {thresholds[0]}, max Threshold: {thresholds[-1]}')
    from sklearn.metrics import average_precision_score, f1_score
    auprc = average_precision_score(label_list, prob_list)
    # F1_score = f1_score(label_list, prob_list, average='micro')
    print(f'AUC: {auc}, SPAUC: {spauc}, AUPRC: {auprc}')
    print(classification_report(label_list, logit_list, target_names=['0', '1']))
    tmp = []
    for _ in range(len(precision)):
        if 0.895 <= precision[_] <= 0.905:
            tmp.append(recall[_])
    if len(tmp):
        print('r@p at 0.9', sorted(list(tmp))[-1])
    

def load_source_domain_representation(src_model_name):
    """
    return src_rep
    """
    with open(src_model_name.replace(".pt", "_rep.pkl"), "rb") as fp:
        data = pickle.load(fp)
        labels = np.array(data["label"])
        reps = data["rep"]
    return reps, labels


def generate_representation(model, dataset):
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
    labs = np.array(label_list)
    return reps, labs


input_size = None
hidden_size = 300
layer_num = 2
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_spauc = 0
latest_update_epoch = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='validate')  # validate
    parser.add_argument('--src_root', default='../Data/LZD')
    parser.add_argument('--tgt_root', default='../Data/HK')
    parser.add_argument('--filepath', default='train_2020-01.csv')
    parser.add_argument('--lr', default=0.0001, type=float)  # 0.001 for LZD
    parser.add_argument('--src_datasets', default='LZD')
    parser.add_argument('--tgt_datasets', default='HK')
    parser.add_argument('--maxepoch', default=2000, type=int)  # 200 for LZD
    parser.add_argument('--loss', default='cross_entropy')
    parser.add_argument('--mark', default="dsan2", type=str)
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
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=0)
    start = 0
    patience = 50
    early_stopping = EarlyStopping(patience, verbose=False, model_name=tgt_model_name)

    
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
        best_spauc = checkpoint["best_spauc"]
        latest_update_epoch = checkpoint["latest_update_epoch"]
        print('exist {}! restart from {}'.format(tgt_model_name, start))

    src_rep, src_label = load_source_domain_representation(src_model_name)
    tgt_rep, tgt_label = generate_representation(model, train_dataset)
    loss_func = TransferLMMDLoss(gamma=float(args.gamma))
    loss_func.update_src_representation(src_rep)
    loss_func.update_tgt_representation(tgt_rep)
    loss_func.update_label(src_label, tgt_label)

    if args.mode == 'train':
        for epoch in range(start, int(args.maxepoch)):
            loss_func.calc_representation_distance()
            train(model, train_loader, optimizer, epoch, loss_func=loss_func)
            val_loss = eval(model, eval_loader, optimizer, epoch, loss_func=loss_func, model_name=tgt_model_name)

            src_hour_rep_list, labels = generate_representation(model, src_dataset)
            loss_func.update_src_representation(src_hour_rep_list)
            tgt_hour_rep_list, labels = generate_representation(model, train_dataset)
            loss_func.update_tgt_representation(tgt_hour_rep_list)

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

    elif args.mode == 'rep':
        model.load_state_dict(checkpoint["model"])
        loader = DataLoader(src_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        label_list = []
        rep_list = []
        model.eval()
        with torch.no_grad():
            with tqdm(enumerate(loader), desc="loading representation...") as loop:
                for i, batch in loop:
                    inputs, labels, lengths = batch
                    rep, _ = model(inputs, lengths)
                    for _ in rep:
                        rep_list.append(_.cpu().numpy())
                    label_list.append(labels.cpu().numpy())
        np.save("./rep/rep_list_data_dsan.npy", np.array(rep_list))
        np.save("./rep/label_list_data_dsan.npy", np.array(label_list)) 
