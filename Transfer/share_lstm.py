"""share the same feature extractor"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter, writer
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import argparse
import joblib


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


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num, finetune=False):
        super(LSTMClassifier, self).__init__()
        self.finetune = finetune
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num)
        self.fc = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.8)

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

        prob = self.dropout(self.fc(representation))

        return representation, prob


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


def train(model, train_loader, optimizer, epoch, loss_func_name='cross_entropy', desc='Train'):
    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    model.train()
    loss_func = FocalLoss() if loss_func_name == 'focal' else nn.CrossEntropyLoss(
        weight=torch.Tensor([0.1, 0.8]).to(device))

    with tqdm(enumerate(train_loader), desc=desc) as loop:
        for i, batch in loop:
            inputs, labels, lengths = batch
            model.zero_grad()
            rep, prob = model(inputs, lengths)
            logits = torch.argmax(prob, dim=-1)

            loss = loss_func(prob, labels)
            loss.backward()
            optimizer.step()

            batch_accuracy = (logits == labels).float().sum().item()
            train_accuracy += batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

    loop.set_postfix(epoch=epoch, loss=train_loss / train_epoch_size, acc=train_accuracy / train_epoch_size)
    writer.add_scalar('loss/train_loss', np.mean(train_loss), epoch)


def eval(model, eval_loader, optimizer, epoch, loss_func_name='cross_entropy', desc='Validation', verbose=True,
         model_name='best_model.pt'):
    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0
    label_list = []
    prob_list = []
    logit_list = []

    model.eval()
    loss_func = FocalLoss() if loss_func_name == 'focal' else nn.CrossEntropyLoss(
        weight=torch.Tensor([0.1, 0.8]).to(device))

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
                validation_loss += loss.item() * batch_size
                loop.set_postfix(epoch=epoch, loss=validation_loss / validation_epoch_size,
                                 acc=validation_accuracy / validation_epoch_size)

    writer.add_scalar('loss/val_loss', np.mean(validation_loss), epoch)

    global best_spauc
    # print("label_list:",type(label_list[-1][0]),label_list[-1])
    # print("prob_list:",type(prob_list[-1][0]),prob_list[-1])
    auc = roc_auc_score(label_list, prob_list)
    spauc = roc_auc_score(label_list, prob_list, max_fpr=0.01)
    if verbose:
        print(f'Epoch {epoch}, Validation AUC: {auc}, Validation SPAUC: {spauc}')
        print(classification_report(label_list, logit_list, target_names=['0', '1']))

    if spauc > best_spauc:
        best_spauc = spauc
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, "best_auc": auc,
                 "best_spauc": spauc}
        torch.save(state, model_name)
        print("Updating model... auc is {}, best spauc is:{}".format(auc, best_spauc))
        # saving prediction results
        with open(model_name + ".prediction.dict", "wb") as fp:
            pickle.dump({"prob_list": prob_list, "logit_list": logit_list}, fp)
    else:
        # update epoch
        checkpoint = torch.load(model_name)
        state = {'model': checkpoint["model"], 'optimizer': checkpoint["optimizer"], 'epoch': epoch,
                 "best_auc": checkpoint["best_auc"], "best_spauc": checkpoint["best_spauc"]}
        torch.save(state, model_name)
        print("Updating epoch... auc is {}, best spauc is:{}".format(checkpoint["best_auc"], checkpoint["best_spauc"]))


def eval_wo_update(model, loader, loss_func_name='cross_entropy', desc='Validation'):
    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0
    label_list = []
    prob_list = []
    logit_list = []
    rep_list = []

    model.eval()
    loss_func = FocalLoss() if loss_func_name == 'focal' else nn.CrossEntropyLoss(
        weight=torch.Tensor([0.1, 0.8]).to(device))

    with torch.no_grad():
        with tqdm(enumerate(loader), desc=desc) as loop:
            for i, batch in loop:
                inputs, labels, lengths = batch
                rep, output = model(inputs, lengths)
                # print("rep",type(rep),rep.shape)
                logits = torch.argmax(output, dim=-1)

                loss = loss_func(output, labels)

                label_list.append(labels.cpu().numpy())
                prob_list.append(torch.softmax(output, dim=-1)[..., 1].cpu().numpy())
                logit_list.append(logits.cpu().numpy())
                rep_list.append(rep)

                batch_accuracy = (logits == labels).float().sum().item()
                validation_accuracy += batch_accuracy
                validation_epoch_size += 1
                validation_loss += loss.item() * batch_size
                loop.set_postfix(loss=validation_loss / validation_epoch_size,
                                 acc=validation_accuracy / validation_epoch_size)

    auc = roc_auc_score(label_list, prob_list)
    spauc = roc_auc_score(label_list, prob_list, max_fpr=0.01)
    print(f'Validation AUC: {auc}, Validation SPAUC: {spauc}')
    print(classification_report(label_list, logit_list, target_names=['0', '1']))


input_size = None
hidden_size = 300
layer_num = 2
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_spauc = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')  # validate
    parser.add_argument('--src_root', default='E:\Transfer_Learning\Data\LZD\csv')
    parser.add_argument('--tgt_root', default='E:\Transfer_Learning\Data\HK\csv')
    parser.add_argument('--filepath', default='train_2020-01.csv')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--src_datasets', default='LZD')
    parser.add_argument('--tgt_datasets', default='HK')
    parser.add_argument('--maxepoch', default=100, type=int)
    parser.add_argument('--loss', default='cross_entropy')
    parser.add_argument('--finetune', default=False)
    parser.add_argument('--mark', default="finetune", type=str)
    args = parser.parse_args()

    src_trainpath = os.path.join(args.src_root, args.filepath)
    src_evalpath = os.path.join(args.src_root, args.filepath.replace("train", "val"))
    src_testpath = os.path.join(args.src_root, args.filepath.replace("train", "test"))
    src_trainOutputName = args.src_datasets + "_" + os.path.basename(src_trainpath).replace(".csv", "") + "(finetune).pkl"
    src_evalOutputName = args.src_datasets + "_" + os.path.basename(src_evalpath).replace(".csv", "") + "(finetune).pkl"
    src_testOutputName = args.src_datasets + "_" + os.path.basename(src_testpath).replace(".csv", "") + "(finetune).pkl"
    src_model_name = args.src_datasets + "_" + src_testpath.split("/")[-1].split("_")[-1].replace(".csv",
                                                                                                  "") + "_" + args.mark + ".pt"
    tgt_trainpath = os.path.join(args.tgt_root, args.filepath)
    tgt_evalpath = os.path.join(args.tgt_root, args.filepath.replace("train", "val"))
    tgt_testpath = os.path.join(args.tgt_root, args.filepath.replace("train", "test"))
    tgt_trainOutputName = args.tgt_datasets + "_" + os.path.basename(tgt_trainpath).replace(".csv", "") + "(finetune).pkl"
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

    if not args.finetune:
        writer = SummaryWriter(src_model_name.replace("pt", ""))
        # loading src train set
        if os.path.exists(src_trainOutputName):
            with open(src_trainOutputName, "rb") as fp:
                train_dataset = pickle.load(fp)
        else:
            train_dataset = EncodedDataset(src_trainpath, tgt_trainpath)
            with open(src_trainOutputName, "wb") as fp:
                pickle.dump(train_dataset, fp)
        # loading src val set
        if os.path.exists(src_evalOutputName):
            with open(src_evalOutputName, "rb") as fp:
                val_dataset = pickle.load(fp)
        else:
            val_dataset = EncodedDataset(src_evalpath, tgt_evalpath)
            with open(src_evalOutputName, "wb") as fp:
                pickle.dump(val_dataset, fp)
        # loading src test set
        if os.path.exists(src_testOutputName):
            with open(src_testOutputName, "rb") as fp:
                test_dataset = pickle.load(fp)
        else:
            test_dataset = EncodedDataset(src_testpath, tgt_testpath)
            with open(src_testOutputName, "wb") as fp:
                pickle.dump(test_dataset, fp)
    else:
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

    model = LSTMClassifier(input_size, hidden_size, layer_num, args.finetune).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=float(args.lr), lr_decay=0, weight_decay=0,
                              initial_accumulator_value=0)
    start = 0
    if not args.finetune:
        if os.path.exists(src_model_name):
            checkpoint = torch.load(src_model_name)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start = checkpoint['epoch'] + 1
            best_spauc = checkpoint["best_spauc"]
            print('exist{}! start from {}'.format(src_model_name, start))
    else:
        if os.path.exists(src_model_name):
            checkpoint = torch.load(src_model_name)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start = checkpoint['epoch'] + 1
            best_spauc = 0
            print('exist{}! start finetune from {}'.format(src_model_name, start))
        else:
            raise FileNotFoundError("initial extractor model not found.")

    if args.mode == 'train':
        for epoch in range(start, int(args.maxepoch)):
            train(model, train_loader, optimizer, epoch, loss_func_name=args.loss)
            if not args.finetune:
                eval(model, eval_loader, optimizer, epoch, loss_func_name=args.loss, model_name=src_model_name)
            else:
                eval(model, eval_loader, optimizer, epoch, loss_func_name=args.loss, model_name=tgt_model_name)
        if not args.finetune:
            checkpoint = torch.load(src_model_name)
        else:
            checkpoint = torch.load(tgt_model_name)
        model.load_state_dict(checkpoint["model"])
        print("evaluating val set...")
        eval_wo_update(model, eval_loader, loss_func_name=args.loss)
        print("evaluating test set...")
        eval_wo_update(model, test_loader, loss_func_name=args.loss)

    elif args.mode == 'validate':
        model.load_state_dict(checkpoint["model"])
        print("evaluating val set...")
        eval_wo_update(model, eval_loader, loss_func_name=args.loss)
        print("evaluating test set...")
        eval_wo_update(model, test_loader, loss_func_name=args.loss)