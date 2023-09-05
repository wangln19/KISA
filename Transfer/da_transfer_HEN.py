import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
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
from scipy.spatial.distance import pdist, squareform
import math
from Earlystopping import EarlyStopping
from Dataset import EncodedDataset
from Model import LSTMClassifier
import time


class DomainAdaptionLoss(nn.Module):
    def __init__(self, src_rep_0, src_rep_1, lamda=0.5):
        super(DomainAdaptionLoss, self).__init__()
        # print("src_rep_0.mean(axis=0)",src_rep_0.mean(axis=0))
        self.src_rep_0 = torch.from_numpy(src_rep_0).cuda(device).reshape((1, -1))
        self.src_rep_1 = torch.from_numpy(src_rep_1).cuda(device).reshape((1, -1))
        self.lamda = lamda  # trade-off parameters
        # print("lamda", self.lamda)
        self.cls = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.8]).to(device))

    def domain_distance(self, rep_1, rep_2):
        return F.pairwise_distance(rep_1, rep_2, p=2)  # 2-order distance

    def update_tgt_representation(self, tgt_rep_0, tgt_rep_1):
        self.tgt_rep_0 = tgt_rep_0.reshape((1, -1))
        self.tgt_rep_1 = tgt_rep_1.reshape((1, -1))

        # print("self.tgt_rep_0:",self.tgt_rep_0.shape)

        # DA loss 分子
        numerator = self.domain_distance(self.src_rep_0, self.tgt_rep_0) + self.domain_distance(self.src_rep_1,
                                                                                                self.tgt_rep_1)
        # DA loss 分母
        denominator = self.domain_distance(self.src_rep_0, self.src_rep_1) + self.domain_distance(self.src_rep_0,
                                                                                                  self.tgt_rep_1) + self.domain_distance(
            self.tgt_rep_0, self.src_rep_1) + self.domain_distance(self.tgt_rep_0, self.tgt_rep_1)

        da_loss = torch.div(numerator, denominator)
        # print("da_loss:", da_loss)
        self.da_loss = da_loss

    def forward(self, prob, labels):
        cls_loss = self.cls(prob, labels)
        return cls_loss + self.lamda * self.da_loss


def generate_representation(model, data_loader):
    loader = DataLoader(data_loader, batch_size=1, shuffle=False, collate_fn=collate_fn)
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

    labels = np.array(label_list)
    indices_0 = np.where(labels == 0)[0]
    indices_1 = np.where(labels == 1)[0]
    reps = torch.stack(rep_list, axis=0).squeeze()
    return reps[indices_0].mean(axis=0), reps[indices_1].mean(axis=0)


def load_source_domain_representation(source_rep_name):
    '''
    return src_rep_0, src_rep_1 shape:(hidden_size, )
    '''
    with open(source_rep_name + "_rep.pkl", "rb") as fp:
        data = pickle.load(fp)
        labels = np.array(data["label"])
        reps = data["rep"]
    indices_0 = np.where(labels == 0)[0]
    indices_1 = np.where(labels == 1)[0]
    return reps[indices_0].mean(axis=0), reps[indices_1].mean(axis=0)


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
    # da_loss = 0
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
                match_loss += (loss_func.lamda * loss_func.da_loss).item()

                loop.set_postfix(epoch=epoch, loss=validation_loss / validation_epoch_size,
                                 acc=validation_accuracy / validation_epoch_size)

    writer.add_scalar('loss/val_loss', np.mean(validation_loss), epoch)
    writer.add_scalar('loss/match_loss', np.mean(match_loss), epoch)
    # writer.add_scalar('loss/da_loss', np.mean(da_loss), epoch)

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
    # print(precision[-20:])


input_size = None
hidden_size = 300
layer_num = 2
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_spauc = 0
latest_update_epoch = 0
num_of_compression = 500


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
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--mark', default="da_transfer_HEN2", type=str)

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

    source_name = "LZD_2020-07_finetune"
    src_rep_0, src_rep_1 = load_source_domain_representation(source_name)
    loss_func = DomainAdaptionLoss(src_rep_0, src_rep_1, lamda=args.gamma)
    tgt_rep_0, tgt_rep_1 = generate_representation(model, train_dataset)
    loss_func.update_tgt_representation(tgt_rep_0, tgt_rep_1)

    if args.mode == 'train':
        for epoch in range(start, int(args.maxepoch)):
            train(model, train_loader, optimizer, epoch, loss_func=loss_func)
            tgt_rep_0, tgt_rep_1 = generate_representation(model, train_dataset)
            loss_func.update_tgt_representation(tgt_rep_0, tgt_rep_1)
            val_loss = eval(model, eval_loader, optimizer, epoch, loss_func=loss_func, model_name=tgt_model_name)

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
        print(f'latest update epoch: {latest_update_epoch}')
        print("evaluating val set...")
        eval_wo_update(model, eval_loader)
        print("evaluating test set...")
        eval_wo_update(model, test_loader)

