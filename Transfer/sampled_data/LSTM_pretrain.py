import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter, writer

from sklearn.metrics import roc_auc_score,classification_report

import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import argparse
import joblib

class EncodedDataset(Dataset):
    def __init__(self, datafile):
        super(EncodedDataset, self).__init__()
        self.data = []
        self.label = []
        self.length = []

        global input_size
        df = pd.read_csv(datafile, encoding='utf-8', sep=',', engine='python', error_bad_lines=False).drop(
            ['apdid', 'routermac'], axis=1)
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


def collate_fn(batch):
    inputs, labels, lengths = zip(*batch)
    inputs_pad = pad_sequence([torch.from_numpy(x) for x in inputs], padding_value=0)
    return inputs_pad.float().to(device), torch.LongTensor(labels).to(device), torch.LongTensor(lengths).to(device)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num)
        self.fc = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x, x_length):
        x = pack_padded_sequence(x, x_length.cpu(), enforce_sorted=False)
        x, hidden = self.lstm(x)
        output, _ = pad_packed_sequence(x)
        seq_len, batch_size, hidden_size = output.shape
        output = output.contiguous().view(batch_size * seq_len, hidden_size)
        output = self.dropout(self.fc(output))
        output = output.view(seq_len, batch_size, -1)
        logits = []
        for i, length in enumerate(x_length):
            logits.append(output[length - 1, i, :])
        return torch.stack(logits, dim=0)


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
    loss_func = FocalLoss() if loss_func_name == 'focal' else nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.8]).to(device))

    with tqdm(enumerate(train_loader), desc=desc) as loop:
        for i, batch in loop:
            inputs, labels, lengths = batch
            model.zero_grad()
            prob = model(inputs, lengths)
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

def eval(model, eval_loader, optimizer, epoch, loss_func_name='cross_entropy', desc='Validation', verbose=True, model_name='best_model.pt'):
    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0
    label_list = []
    prob_list = []
    logit_list = []

    model.eval()
    loss_func = FocalLoss() if loss_func_name == 'focal' else nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.8]).to(device))

    with torch.no_grad():
        with tqdm(enumerate(eval_loader), desc=desc) as loop:
            for i, batch in loop:
                inputs, labels, lengths = batch
                output = model(inputs, lengths)
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

    global best_auc
    #print("label_list:",type(label_list[-1][0]),label_list[-1])
    #print("prob_list:",type(prob_list[-1][0]),prob_list[-1])
    auc = roc_auc_score(label_list, prob_list)
    spauc = roc_auc_score(label_list, prob_list,max_fpr=0.01)
    if verbose:
        print(f'Epoch {epoch}, Validation AUC: {auc}, Validation SPAUC: {spauc}')
        print(classification_report(label_list, logit_list, target_names=['0', '1']))

    if auc > best_auc:
        best_auc = auc
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, "best_auc": best_auc,"best_spauc":spauc}
        torch.save(state, model_name)
        print("Updating model... spauc is {}, best auc is:{}".format(spauc,best_auc))
        # saving prediction results
        with open(model_name + ".prediction.dict","wb") as fp:
            pickle.dump({"prob_list":prob_list,"logit_list":logit_list},fp)
    else:
        # update epoch
        checkpoint = torch.load(model_name)
        state = {'model': checkpoint["model"], 'optimizer': checkpoint["optimizer"], 'epoch': epoch, "best_auc": checkpoint["best_auc"],"best_spauc":checkpoint["best_spauc"]}
        torch.save(state, model_name)
        print("Updating epoch... best spauc is {}, auc is:{}".format(checkpoint["best_spauc"],checkpoint["best_auc"]))
    
def eval_wo_update(model, loader, epoch, loss_func_name='cross_entropy', desc='Validation', verbose=True, model_name='best_model.pt'):
    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0
    label_list = []
    prob_list = []
    logit_list = []

    model.eval()
    loss_func = FocalLoss() if loss_func_name == 'focal' else nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.8]).to(device))

    with torch.no_grad():
        with tqdm(enumerate(loader), desc=desc) as loop:
            for i, batch in loop:
                inputs, labels, lengths = batch
                output = model(inputs, lengths)
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


    #print("label_list:",type(label_list[-1][0]),label_list[-1])
    #print("prob_list:",type(prob_list[-1][0]),prob_list[-1])
    auc = roc_auc_score(label_list, prob_list)
    spauc = roc_auc_score(label_list, prob_list,max_fpr=0.01)
    print(f'Epoch {epoch}, Validation AUC: {auc}, Validation SPAUC: {spauc}')
    print(classification_report(label_list, logit_list, target_names=['0', '1']))

    

input_size = None
hidden_size = 300
layer_num = 2
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_auc = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')  # validate
    parser.add_argument('--baseroot', default='E:/root/ChenLiyue/ant/sliding_data/LZD/csv')
    parser.add_argument('--filepath', default='train_2020-01.csv')
    parser.add_argument('--datasets', default='LZD')
    parser.add_argument('--maxepoch', default=30, type=int)
    parser.add_argument('--loss', default='cross_entropy')
    # parser.add_argument('--filepath', default='./data/HK.pkl')
   
    parser.add_argument('--mark', default="lstm", type=str)
    args = parser.parse_args()

    trainpath = os.path.join(args.baseroot,args.filepath)
    evalpath = os.path.join(args.baseroot,args.filepath.replace("train","val"))
    testpath = os.path.join(args.baseroot,args.filepath.replace("train","test"))
    trainOutputName = args.datasets + "_" + os.path.basename(trainpath).replace(".csv","") + "_" + args.mark + ".pkl"
    evalOutputName = args.datasets + "_" + os.path.basename(evalpath).replace(".csv","") + "_" + args.mark + ".pkl"
    testOutputName = args.datasets + "_" + os.path.basename(testpath).replace(".csv","") + "_" + args.mark + ".pkl"
    model_name = args.datasets + "_" + testpath.split("/")[-1].split("_")[-1].replace(".csv","") + "_" + args.mark + ".pt"
    writer = SummaryWriter(model_name.replace("pt",""))
    print("---------------------------------------------------")
    print("train output name:", trainOutputName)
    print("val output name:", evalOutputName)
    print("test output name:", testOutputName)
    print("model name:", model_name)
    print("device",device)
    print("---------------------------------------------------")

    # loading train set
    if os.path.exists(trainOutputName):
        with open(trainOutputName,"rb") as fp:
            train_dataset = pickle.load(fp)
    else:
        train_dataset = EncodedDataset(trainpath)
        with open(trainOutputName, "wb") as fp:
            pickle.dump(train_dataset,fp)

    # loading val set
    if os.path.exists(evalOutputName):
        with open(evalOutputName,"rb") as fp:
            val_dataset = pickle.load(fp)
    else:
        val_dataset = EncodedDataset(evalpath)
        with open(evalOutputName, "wb") as fp:
            pickle.dump(val_dataset,fp)
    
    # loading test set
    if os.path.exists(testOutputName):
        with open(testOutputName,"rb") as fp:
            test_dataset = pickle.load(fp)
    else:
        test_dataset = EncodedDataset(testpath)
        with open(testOutputName, "wb") as fp:
            pickle.dump(test_dataset,fp)


    input_size = train_dataset[0][0].shape[1]
    
    writer = SummaryWriter(model_name.replace("pt",""))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = LSTMClassifier(input_size, hidden_size, layer_num).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=0.001, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    start = 0

    if os.path.exists(model_name):
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
        best_auc = checkpoint["best_auc"]

    if args.mode == 'train':
        for epoch in range(start, int(args.maxepoch)):
            train(model, train_loader, optimizer, epoch, loss_func_name=args.loss)
            eval(model, eval_loader, optimizer, epoch, loss_func_name=args.loss, model_name=model_name)

        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint["model"])
        print("evaluating val set...")
        eval_wo_update(model, eval_loader, start, loss_func_name=args.loss, verbose=True, model_name=model_name)
        print("evaluating test set...")
        eval_wo_update(model, test_loader, start, loss_func_name=args.loss, verbose=True, model_name=model_name)

    elif args.mode == 'validate':
        model.load_state_dict(checkpoint["model"])
        eval_wo_update(model, eval_loader, start, loss_func_name=args.loss, verbose=True, model_name=model_name)
        print("evaluating test set...")
        eval_wo_update(model, test_loader, start, loss_func_name=args.loss, verbose=True, model_name=model_name)