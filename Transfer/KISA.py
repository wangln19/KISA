import imp
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
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
from Dataset import FusionDataset
from Loss import MMD_Loss
import time
import random


def collate_fn(batch):
    rep1, rep2, label = zip(*batch)
    return torch.Tensor(rep1).to(device), torch.Tensor(rep2).to(device), torch.LongTensor(label).to(device)


class FusionClassifier(nn.Module):
    def __init__(self, hidden_size):
        super(FusionClassifier, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 1)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.fc3 = nn.Linear(hidden_size, 128)
        self.fc4 = nn.Linear(128, 16)
        self.fc5 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(0.8)

    def forward(self, rep1, rep2):
        weight1 = torch.sigmoid(self.fc1(rep1))
        weight2 = torch.sigmoid(self.fc2(rep2))
        x = weight1 / (weight1 + weight2) * rep1 + weight2 / (weight1 + weight2) * rep2
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        output = self.dropout(torch.sigmoid(self.fc5(x)))
        return output


def train(model, train_loader, optimizer, epoch, loss_func=nn.CrossEntropyLoss(), desc='Train'):
    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    model.train()
    with tqdm(enumerate(train_loader), desc=desc) as loop:
        for i, batch in loop:
            rep1, rep2, label = batch
            model.zero_grad()
            prob = model(rep1, rep2)
            logits = torch.argmax(prob, dim=-1)
            label = label.squeeze()
            loss = loss_func(prob, label)
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            batch_accuracy = (logits == label).float().sum().item()
            train_accuracy += batch_accuracy
            b_size = rep1.shape[0]
            train_epoch_size += b_size
            train_loss += loss.item() * b_size

    loop.set_postfix(epoch=epoch, loss=train_loss / train_epoch_size, acc=train_accuracy / train_epoch_size)
    writer.add_scalar('loss/train_loss', train_loss, epoch)


def eval(model, eval_loader, epoch, loss_func=nn.CrossEntropyLoss(), desc='Validation'):
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
                rep1, rep2, label = batch
                output = model(rep1, rep2)
                label = label.squeeze()
                logits = torch.argmax(output, dim=-1)

                loss = loss_func(output, label.view(-1))
                label_list.append(label.cpu().numpy())
                prob_list.append(torch.softmax(output, dim=-1)[..., 1].cpu().numpy())
                logit_list.append(logits.cpu().numpy())

                batch_accuracy = (logits == label).float().sum().item()
                validation_accuracy += batch_accuracy
                validation_epoch_size += 1
                validation_loss += loss.item()

                loop.set_postfix(epoch=epoch, loss=validation_loss / validation_epoch_size,
                                 acc=validation_accuracy / validation_epoch_size)

    writer.add_scalar('loss/val_loss', validation_loss, epoch)

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
    return validation_loss


def eval_wo_update(model, loader, desc='Validation', save_rep=False):
    validation_accuracy = 0
    validation_epoch_size = 0
    label_list = []
    prob_list = []
    logit_list = []

    model.eval()

    with torch.no_grad():
        with tqdm(enumerate(loader), desc=desc) as loop:
            for i, batch in loop:
                rep1, rep2, label = batch
                output = model(rep1, rep2)
                # print("rep",type(rep),rep.shape)
                logits = torch.argmax(output, dim=-1)

                label_list.append(label.cpu().numpy())
                prob_list.append(torch.softmax(output, dim=-1)[..., 1].cpu().numpy())
                logit_list.append(logits.cpu().numpy())

                batch_accuracy = (logits == label).float().sum().item()
                validation_accuracy += batch_accuracy
                validation_epoch_size += 1

                loop.set_postfix(acc=validation_accuracy / validation_epoch_size)

    
    label_list = np.array(label_list).squeeze()
    prob_list = np.array(prob_list).squeeze()
    auc = roc_auc_score(label_list, prob_list)
    spauc = roc_auc_score(label_list, prob_list, max_fpr=0.1)
    precision, recall, thresholds = precision_recall_curve(label_list, prob_list)
    
    print(f'min Threshold: {thresholds[0]}, max Threshold: {thresholds[-1]}')
    from sklearn.metrics import average_precision_score, f1_score
    auprc = average_precision_score(label_list, prob_list)
    # F1_score = f1_score(label_list, prob_list, average='micro')
    print(f'AUC: {auc}, SPAUC: {spauc}, AUPRC: {auprc}')
    print(classification_report(label_list, logit_list, target_names=['0', '1']))


hidden_size = 300
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')  # validate
    parser.add_argument('--dataset', default='HK_2020-03')
    parser.add_argument('--lr', default=0.01, type=float)  # 0.001 
    parser.add_argument('--maxepoch', default=10000, type=int)  # 200 for LZD
    parser.add_argument('--loss', default='cross_entropy')
    parser.add_argument('--mark', default="KISA", type=str)
    args = parser.parse_args()

    hour_train_rep_name = args.dataset + '_' + args.mark + '_hour_train_rep.pkl'
    hour_val_rep_name = args.dataset + '_' + args.mark + '_hour_val_rep.pkl'
    hour_test_rep_name = args.dataset + '_' + args.mark + '_hour_test_rep.pkl'
    type_train_rep_name = args.dataset + '_' + args.mark + '_type_train_rep.pkl'
    type_val_rep_name = args.dataset + '_' + args.mark + '_type_val_rep.pkl'
    type_test_rep_name = args.dataset + '_' + args.mark + '_type_test_rep.pkl'
    model_name = args.dataset + '_' + args.mark + '.pt'
    print("---------------------------------------------------")
    print("hour train name:", hour_train_rep_name)
    print("hour val name:", hour_val_rep_name)
    print("hour test name:", hour_test_rep_name)
    print("type train name:", type_train_rep_name)
    print("type val name:", type_val_rep_name)
    print("type test name:", type_test_rep_name)
    print("model name:", model_name)
    print("device", device)
    print("---------------------------------------------------")

    writer = SummaryWriter(model_name.replace(".pt", ""))
    train_dataset = FusionDataset(hour_train_rep_name, type_train_rep_name)
    val_dataset = FusionDataset(hour_val_rep_name, type_val_rep_name)
    test_dataset = FusionDataset(hour_test_rep_name, type_test_rep_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = FusionClassifier(hidden_size).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=float(args.lr), lr_decay=0, weight_decay=0,
                              initial_accumulator_value=0)

    start = 0
    patience = 20
    early_stopping = EarlyStopping(patience, verbose=False, model_name=model_name)
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

    if os.path.exists(model_name):
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
        latest_update_epoch = checkpoint["latest_update_epoch"]
        print('exist {}! restart from {}'.format(model_name, start))

    if args.mode == 'train':
        for epoch in range(start, int(args.maxepoch)):
            train(model, train_loader, optimizer, epoch)
            val_loss = eval(model, eval_loader, epoch)

            if epoch > 10:
                early_stopping(val_loss, model, optimizer)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint["model"])
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
    
    elif args.mode == 'generate':
        model.load_state_dict(checkpoint["model"])
        gen_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        print("generating representation...")
        eval_wo_update(model, gen_loader, save_rep=True)
