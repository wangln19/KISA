"""
Loss funcs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random


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


class TransferMeanLoss(nn.Module):
    def __init__(self, gamma=0.001):
        super(TransferMeanLoss, self).__init__()
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

    def update_src_representation(self, src_hour_list, src_hour_rep_list):
        src_hour_centroid_list = []
        for _ in range(len(src_hour_rep_list)):
            src_hour_centroid_list.append(src_hour_rep_list[_].mean(axis=0))
        self.src_num_spaces = len(src_hour_list)
        self.rep_hidden_states = len(src_hour_centroid_list[0])
        if isinstance(src_hour_centroid_list[0], torch.Tensor):
            self.src_hour_centroid_list = [item.reshape((1, -1)) for item in src_hour_centroid_list]
        else:
            self.src_hour_centroid_list = [torch.from_numpy(item).cuda(device).reshape((1, -1)) for item in
                                        src_hour_centroid_list]
        self.src_hour_centroid_matrix = torch.cat(self.src_hour_centroid_list, axis=0)  # (src_num_spaces, hidden_states)
        # print("self.src_hour_centroid_matrix:", self.src_hour_centroid_matrix.shape)

    def update_tgt_representation(self, model, tgt_hour_list='default', tgt_hour_rep_list='default'):
        if tgt_hour_list is not 'default':
            self.tgt_num_space = len(tgt_hour_list)
        if tgt_hour_rep_list is not 'default':
            tgt_hour_centroid_list = []
            for _ in range(len(tgt_hour_rep_list)):
                tgt_hour_centroid_list.append(tgt_hour_rep_list[_].mean(axis=0))
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

    def forward(self, prob, labels, model):
        cls_loss = self.cls(prob, labels)
        self.update_tgt_representation(model)
        return cls_loss + self.gamma * self.match_loss


def Guassian_Kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
        source: 源域数据（n * len(x))
        target: 目标域数据（m * len(y))
        kernel_mul:
        kernel_num: 取不同高斯核的数量
        fix_sigma: 不同高斯核的sigma值
    Return:
        sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def MMD_Loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
        source: 源域数据（n * len(x))
        target: 目标域数据（m * len(y))
        kernel_mul:
        kernel_num: 取不同高斯核的数量
        fix_sigma: 不同高斯核的sigma值
    Return:
        loss: MMD loss
    '''
    source, target = torch.Tensor(source), target.cpu()
    source, target = Variable(source), Variable(target)
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = Guassian_Kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return float(loss)  # 因为一般都是n==m，所以L矩阵一般不加入计算


class TransferMMDLoss(nn.Module):
    def __init__(self, gamma=0.001):
        super(TransferMMDLoss, self).__init__()
        self.gamma = gamma  # trade-off parameters
        # print("gamma", self.gamma)
        self.cls = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.8]).to(device))

    def update_src_representation(self, src_hour_list, src_hour_rep_list):
        self.src_num_space = len(src_hour_list)
        self.rep_hidden_states = src_hour_rep_list[0].shape[1]
        self.src_hour_rep_list = src_hour_rep_list

    def update_tgt_representation(self, tgt_hour_list, tgt_hour_rep_list):
        self.tgt_num_space = len(tgt_hour_list)
        self.tgt_hour_rep_list = tgt_hour_rep_list

    def select_src_representation(self, num):
        slt_src_rep_list = []
        for _ in range(self.src_num_space):
            count = 0
            if isinstance(self.src_hour_rep_list[_], torch.Tensor):
                self.src_hour_rep_list[_] = self.src_hour_rep_list[_].cpu().numpy()
            while True:
                indices = random.sample(range(self.src_hour_rep_list[_].shape[0]), num)
                selected_centroid = self.src_hour_rep_list[_][indices].mean(axis=0)
                ori_centriod = self.src_hour_rep_list[_].mean(axis=0)
                dist = F.pairwise_distance(torch.from_numpy(ori_centriod.reshape(1,self.rep_hidden_states)),
                                           torch.from_numpy(selected_centroid.reshape(1,self.rep_hidden_states)), p=2)
                dist = float(dist)
                # set select bar = 0.1
                if dist < 0.1:
                    break
                else:
                    count += 1
                    if count > 50:
                        raise RuntimeError("Select Centroid Bar is too Strict.")
            slt_src_rep_list.append(self.src_hour_rep_list[_][indices])
        return slt_src_rep_list

    def calc_representation_distance(self):
        dists = []
        for _ in range(self.tgt_num_space):
            dist = []
            slt_src_rep_list = self.select_src_representation(len(self.tgt_hour_rep_list[_]))
            for __ in range(self.src_num_space):
                dist.append(MMD_Loss(slt_src_rep_list[__], self.tgt_hour_rep_list[_]))
            dists.append(dist)
        distance = np.array(dists)  # (self.tgt_num_space, self.src_num_spaces)
        return distance

    def match_representation(self, top_k=1):
        dist_matrix = self.calc_representation_distance()
        numerator_matrix = np.zeros((self.tgt_num_space, self.src_num_space))
        denominator_matrix = np.ones((self.tgt_num_space, self.src_num_space))
        for _ in range(self.tgt_num_space):
            tgt_list = list(dist_matrix[_])
            tgt_list.sort()
            for __ in range(self.src_num_space):
                if dist_matrix[_, __] in tgt_list[:top_k]:
                    numerator_matrix[_, __] = 1
                    denominator_matrix[_, __] = 0
        return dist_matrix, numerator_matrix, denominator_matrix

    def calculate_match_loss(self):
        dist_matrix, numerator_matrix, denominator_matrix = self.match_representation(top_k=1)
        numerator = torch.mul(torch.from_numpy(dist_matrix), torch.from_numpy(numerator_matrix)).sum().sum()
        denominator = torch.mul(torch.from_numpy(dist_matrix), torch.from_numpy(denominator_matrix)).sum().sum()
        match_loss = numerator / denominator
        self.match_loss = match_loss

    def forward(self, prob, labels):
        cls_loss = self.cls(prob, labels)
        return cls_loss + self.gamma * self.match_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

