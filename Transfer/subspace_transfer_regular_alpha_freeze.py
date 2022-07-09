import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.serialization import load
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
torch.backends.cudnn.enabled = False

torch.autograd.set_detect_anomaly(True)

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
        output = output.view(seq_len, batch_size, -1)
        
        representation = []
        for i, length in enumerate(x_length):
            representation.append(output[length - 1, i, :])
        representation = torch.stack(representation, dim=0)

        return representation, self.dropout(self.fc(representation))

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
    def __init__(self, src_month_list, src_month_centroid_list, gamma=0.001, max_weight= 0.8):
        super(TransferLoss, self).__init__() 
        self.src_num_spaces = len(src_month_list)
        self.src_month_list = src_month_list
        self.rep_hidden_states = len(src_month_centroid_list[0])
        self.src_month_centroid_list = [torch.from_numpy(item).cuda(device).reshape((1,-1)) for item in src_month_centroid_list]
        self.src_month_centroid_matrix = torch.cat(self.src_month_centroid_list, axis=0) # (src_num_spaces, hidden_states)

        self.max_weight = torch.tensor(max_weight).cuda(device=device)
        print("max_weight",self.max_weight)
        # print("self.src_month_centroid_matrix:", self.src_month_centroid_matrix.shape)
        self.gamma = gamma # trade-off parameters
        print("gamma",self.gamma)
        self.cls = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.8]).to(device))
    
    def calc_representation_distance(self):
        distance = torch.zeros((self.src_num_spaces,self.tgt_num_space)).cuda(device)
        for i in range(self.src_num_spaces):
            for j in range(i, self.tgt_num_space):
                distance[i,j] = F.pairwise_distance(self.src_month_centroid_matrix[i:i+1], self.tgt_month_centroid_matrix[j:j+1], p=2)
                distance[j,i] = distance[i,j]
        return distance 
    
    def update_tgt_representation(self, tgt_month_list, tgt_month_centroid_list, model):
        self.tgt_num_space = len(tgt_month_list)
        self.tgt_month_list = tgt_month_list
        self.tgt_month_centroid_matrix = torch.stack(tgt_month_centroid_list, axis=0) # (tgt_num_spaces, hidden_states)
        # print("self.src_month_centroid_matrix:", self.src_month_centroid_matrix)
        # print("self.tgt_month_centroid_matrix:", self.tgt_month_centroid_matrix)

        src_after_trans = torch.mm(self.src_month_centroid_matrix, model.src_weight) # centroid representation in src domain
        tgt_after_trans = torch.mm(self.tgt_month_centroid_matrix, model.tgt_weight) # centroid representation in tgt domain

        # src_after_trans = model.src_weight(self.src_month_centroid_matrix)
        # tgt_after_trans = model.tgt_weight(self.tgt_month_centroid_matrix)
        
        dot_product= torch.mm(src_after_trans,tgt_after_trans.T) /  np.sqrt(self.rep_hidden_states)
        alpha = F.softmax(dot_product, dim=0)
        alpha_regularization_loss = F.relu(alpha - self.max_weight)
        self.alpha_regularization_loss = alpha_regularization_loss.sum().sum()

        distance = self.calc_representation_distance()

        print("alpha:", alpha)
        print("distance:", distance)
        # print("alpha:", alpha.device)
        # print("distance:", distance.device)

        match_loss = torch.mul(alpha, distance).sum().sum()
        # print("match_loss:", match_loss)
        self.match_loss = match_loss

    def forward(self, prob, labels, tgt_month_list, tgt_month_centroid_list, model): 
        cls_loss = self.cls(prob, labels)
        self.update_tgt_representation(tgt_month_list, tgt_month_centroid_list, model)
        print("cls_loss:",cls_loss)
        print("match_loss:",self.gamma*self.match_loss)
        print("alpha_regularization_loss:",self.alpha_regularization_loss)

        # return cls_loss + self.gamma * self.match_loss + self.alpha_regularization_loss
        return cls_loss  + self.alpha_regularization_loss


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


def train(model, train_loader, optimizer, epoch, tgt_month_list, tgt_month_centroid_list, loss_func=nn.CrossEntropyLoss, desc='Train'):
    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    model.train()
    
    with tqdm(enumerate(train_loader), desc=desc) as loop:
        for i, batch in loop:
            inputs, labels, lengths = batch
            model.zero_grad()
            representation, prob = model(inputs, lengths)
            logits = torch.argmax(prob, dim=-1)

            loss = loss_func(prob, labels, tgt_month_list, tgt_month_centroid_list, model)
            # print('before backward ---------------------------------------')
            # print(model.lstm.weight_hh_l0.grad)
            # loss.backward(retain_graph=True)
            loss.backward(retain_graph=True)
   
            # print('after backward ---------------------------------------')
            # print(model.lstm.weight_hh_l0.grad)
            optimizer.step()

            batch_accuracy = (logits == labels).float().sum().item()
            train_accuracy += batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

            loop.set_postfix(epoch=epoch, loss=train_loss / train_epoch_size, acc=train_accuracy / train_epoch_size)
    writer.add_scalar('loss/train_loss', np.mean(train_loss), epoch)

def eval(model, eval_loader, optimizer, epoch, tgt_month_list, tgt_month_centroid_list, loss_func=nn.CrossEntropyLoss, desc='Validation', verbose=True, model_name='best_model.pt'):
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
                representation, output = model(inputs, lengths)
                logits = torch.argmax(output, dim=-1)

                loss = loss_func(output, labels, tgt_month_list, tgt_month_centroid_list, model)

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
    
def eval_wo_update(model, loader, epoch, desc='Validation', verbose=True, model_name='best_model.pt'):
    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0
    label_list = []
    prob_list = []
    logit_list = []

    model.eval()
    
    with torch.no_grad():
        with tqdm(enumerate(loader), desc=desc) as loop:
            for i, batch in loop:
                inputs, labels, lengths = batch
                representation, output = model(inputs, lengths)
                logits = torch.argmax(output, dim=-1)

                
                label_list.append(labels.cpu().numpy())
                prob_list.append(torch.softmax(output, dim=-1)[..., 1].cpu().numpy())
                logit_list.append(logits.cpu().numpy())

                batch_accuracy = (logits == labels).float().sum().item()
                validation_accuracy += batch_accuracy
                validation_epoch_size += 1
                loop.set_postfix(epoch=epoch, acc=validation_accuracy / validation_epoch_size)

    #print("label_list:",type(label_list[-1][0]),label_list[-1])
    #print("prob_list:",type(prob_list[-1][0]),prob_list[-1])
    auc = roc_auc_score(label_list, prob_list)
    spauc = roc_auc_score(label_list, prob_list,max_fpr=0.01)
    print(f'Epoch {epoch}, Validation AUC: {auc}, Validation SPAUC: {spauc}')
    print(classification_report(label_list, logit_list, target_names=['0', '1']))

def generate_tgt_representation(model, data_loader, target_month_name):
    loader = DataLoader(data_loader, batch_size=1, shuffle=False, collate_fn=collate_fn)
    label_list = []
    rep_list = []
    model.eval()
    with torch.no_grad():
        with tqdm(enumerate(loader),desc="loading representation...") as loop:
            for i, batch in loop:
                inputs, labels, lengths = batch
                rep, _ = model(inputs, lengths)
                rep_list.append(rep)
                label_list.append(labels.cpu().numpy())
    
    reps = torch.stack(rep_list,axis=0).squeeze()
    # labels = np.array(label_list)   

    with open(target_month_name + "_month.pkl","rb") as fp:
        month_indicator = pickle.load(fp)    

    month_list = np.unique(month_indicator)
    month_centroid_list = []
    for mon in month_list:
        indices = np.where(month_indicator==mon)[0]
        month_centroid_list.append(reps[indices].mean(axis=0))         
    return month_list, month_centroid_list           

def load_source_domain_representation(source_rep_name):
    '''
    return src_rep_0, src_rep_1
    '''
    with open(source_rep_name + "_rep.pkl","rb") as fp:
        data = pickle.load(fp)
        labels = np.array(data["label"])
        reps = data["rep"]
    with open(source_rep_name.replace("lstm","month.pkl"),"rb") as fp:
        month_indicator = pickle.load(fp)
    
    month_list = np.unique(month_indicator)
    month_centroid_list = []
    for mon in month_list:
        indices = np.where(month_indicator==mon)[0]
        month_centroid_list.append(reps[indices].mean(axis=0))

    return month_list, month_centroid_list, labels           

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
    parser.add_argument('--lr', default=0.001,type=float)
    parser.add_argument('--datasets', default='LZD')
    parser.add_argument('--maxepoch', default=30, type=int)
    parser.add_argument('--gamma', default=0.001, type=float)
    parser.add_argument('--loss', default='cross_entropy')
    # parser.add_argument('--filepath', default='./data/HK.pkl')
   
    parser.add_argument('--mark', default="alpha_regular", type=str)
    args = parser.parse_args()

    trainpath = os.path.join(args.baseroot,args.filepath)
    evalpath = os.path.join(args.baseroot,args.filepath.replace("train","val"))
    testpath = os.path.join(args.baseroot,args.filepath.replace("train","test"))
    trainOutputName = args.datasets + "_" + os.path.basename(trainpath).replace(".csv","") + ".pkl"
    evalOutputName = args.datasets + "_" + os.path.basename(evalpath).replace(".csv","") + ".pkl"
    testOutputName = args.datasets + "_" + os.path.basename(testpath).replace(".csv","") + ".pkl"
    model_name = args.datasets + "_" + testpath.split("/")[-1].split("_")[-1].replace(".csv","") + "_" + args.mark + ".pt"
    da_model_name = args.datasets + "_" + testpath.split("/")[-1].split("_")[-1].replace(".csv","") + "_da" + ".pt"
    writer = SummaryWriter(model_name.replace("pt",""))
    print("---------------------------------------------------")
    print("train output name:", trainOutputName)
    print("val output name:", evalOutputName)
    print("test output name:", testOutputName)
    print("model name:", model_name)
    print("domain adaption model name:", da_model_name)
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

    # model = LSTMClassifier(input_size, hidden_size, layer_num).to(device)
    model = LSTMTransfer(input_size, hidden_size, layer_num).to(device)
    #optimizer = optim.Adagrad(model.parameters(), lr=float(args.lr), lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

    if os.path.exists(da_model_name):
        print("loading model affer domain adaption...")
        # 使用DA模型中的参数更新现有的 model_dict
        da_checkpoint = torch.load(da_model_name)
        da_model = da_checkpoint['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in da_model.items() if k in model_dict}
        model_dict.update(pretrained_dict)   
        model.load_state_dict(model_dict) 
    else:
        raise FileNotFoundError("domain adaption model not found.")

    if os.path.exists(model_name):
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
        best_auc = checkpoint["best_auc"]
    else:
        start = 0
        best_auc = 0

    def freeze_model_except_name_list(model, except_name_list):
        for name, param in model.named_parameters():
            if name not in except_name_list:
                param.requires_grad = False

    except_name_list = ['src_weight', 'tgt_weight']
    freeze_model_except_name_list(model, except_name_list)

    for name, param in model.named_parameters():
        print("name:",name)
        print("param.requires_grad :",param.requires_grad)

    source_model_name = "LZD_2020-01_lstm"
    target_month_name = args.datasets + "_" + testpath.split("/")[-1].split("_")[-1].replace(".csv","")
    print("target_month_name:",target_month_name)
    src_month_list, src_month_centroid_list, src_labels = load_source_domain_representation(source_model_name)
    tgt_month_list, tgt_month_centroid_list = generate_tgt_representation(model, train_dataset, target_month_name)
    # loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.8]).to(device))
    loss_func = TransferLoss(src_month_list, src_month_centroid_list, gamma= float(args.gamma))
    #loss_func.update_tgt_representation(tgt_month_list, tgt_month_centroid_list, model)


    if args.mode == 'train':
        for epoch in range(start, int(args.maxepoch)):
            train(model, train_loader, optimizer, epoch, tgt_month_list, tgt_month_centroid_list, loss_func=loss_func)
            #loss_func.update_tgt_representation(tgt_month_list, tgt_month_centroid_list, model)
            eval(model, eval_loader, optimizer, epoch, tgt_month_list, tgt_month_centroid_list, loss_func=loss_func, model_name=model_name)
            tgt_month_list, tgt_month_centroid_list = generate_tgt_representation(model, train_dataset, target_month_name)

        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint["model"])
        print("evaluating val set...")
        eval_wo_update(model, eval_loader, epoch, verbose=True, model_name=model_name)
        print("evaluating test set...")
        eval_wo_update(model, test_loader, epoch, verbose=True, model_name=model_name)

    elif args.mode == 'validate':
        model.load_state_dict(checkpoint["model"])
        eval_wo_update(model, eval_loader, start, verbose=True, model_name=model_name)
        print("evaluating test set...")
        eval_wo_update(model, test_loader, start, verbose=True, model_name=model_name)