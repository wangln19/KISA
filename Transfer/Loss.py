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


def coral_loss(source, target):
    """
    input:
    source, target: cuda tensor
    output:
    float
    """
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).to(device) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(device) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss


class TransferCoralLoss(nn.Module):
    def __init__(self, gamma=0.001):
        super(TransferCoralLoss, self).__init__()
        self.gamma = gamma  # trade-off parameters
        # print("gamma", self.gamma)
        self.cls = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.8]).to(device))

    def update_src_representation(self, src_hour_rep):
        self.src_hour_rep = src_hour_rep

    def update_tgt_representation(self, tgt_hour_rep):
        self.tgt_hour_rep = tgt_hour_rep

    def calculate_match_loss(self):
        if isinstance(self.src_hour_rep, np.ndarray):
            self.src_hour_rep = torch.from_numpy(self.src_hour_rep).cuda()
        if isinstance(self.tgt_hour_rep, np.ndarray):
            self.tgt_hour_rep = torch.from_numpy(self.tgt_hour_rep).cuda()
        match_loss = coral_loss(self.src_hour_rep, self.tgt_hour_rep)
        self.match_loss = match_loss

    def forward(self, prob, labels):
        cls_loss = self.cls(prob, labels)
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
    source, target = torch.Tensor(source), torch.Tensor(target)
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

    def select_src_representation(self, num, index):
        count = 0
        src_hour_rep = self.src_hour_rep_list[index]
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
        return slt_src_hour_rep

    def select_tgt_representation(self, num, index):
        count = 0
        tgt_hour_rep = self.tgt_hour_rep_list[index]
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
        return slt_tgt_hour_rep

    def calc_representation_distance(self):
        dists = []
        for _ in range(self.tgt_num_space):
            dist = []
            for __ in range(self.src_num_space):
                if len(self.tgt_hour_rep_list[_]) <= len(self.src_hour_rep_list[__]):
                    slt_src_rep = self.select_src_representation(len(self.tgt_hour_rep_list[_]), __)
                    if isinstance(self.tgt_hour_rep_list[_], torch.Tensor):
                        self.tgt_hour_rep_list[_] = self.tgt_hour_rep_list[_].cpu().numpy()
                    dist.append(MMD_Loss(slt_src_rep, self.tgt_hour_rep_list[_]))
                else:
                    slt_tgt_rep = self.select_tgt_representation(len(self.src_hour_rep_list[__]), _)
                    if isinstance(self.src_hour_rep_list[__], torch.Tensor):
                        self.src_hour_rep_list[__] = self.src_hour_rep_list[__].cpu().numpy()
                    dist.append(MMD_Loss(self.src_hour_rep_list[__], slt_tgt_rep))
            dists.append(dist)
        distance = np.array(dists)  # (self.tgt_num_space, self.src_num_spaces)
        return distance

    def match_representation(self, top_k):
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

    def calculate_match_loss(self, top_k=1):
        dist_matrix, numerator_matrix, denominator_matrix = self.match_representation(top_k)
        numerator = torch.mul(torch.from_numpy(dist_matrix), torch.from_numpy(numerator_matrix)).sum().sum()
        denominator = torch.mul(torch.from_numpy(dist_matrix), torch.from_numpy(denominator_matrix)).sum().sum()
        match_loss = np.log(numerator / denominator)
        self.match_loss = match_loss

    def forward(self, prob, labels):
        cls_loss = self.cls(prob, labels)
        return cls_loss + self.gamma * self.match_loss


class LMMD_loss(nn.Module):
    """
    input: tensor(cuda)
            reps: [n, hid_size]
            label: [1, n]
    output: tensor float(cuda)
    """
    def __init__(self, class_num=2, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=31):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        # t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        # t_vec_label = t_label.cpu().data.numpy()
        t_sca_label = t_label.cpu().data.numpy()
        t_vec_label = self.convert_to_onehot(t_sca_label, class_num=self.class_num)
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


class TransferMeanLoss(nn.Module):
    def __init__(self, gamma=0.5, da_gamma=1):
        super(TransferMeanLoss, self).__init__()
        self.gamma = gamma  # trade-off parameters
        self.da_gamma = da_gamma
        # print("gamma", self.gamma)
        self.cls = nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 0.8]).to(device))

    def update_src_representation(self, src_hour_list, src_hour_rep_list):  # src_rep_0, src_rep_1
        self.src_num_space = len(src_hour_list)
        self.rep_hidden_states = src_hour_rep_list[0].shape[1]
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
                if isinstance(self.src_hour_rep_list[__], np.ndarray):
                    src_hour_rep_centroid = torch.from_numpy(self.src_hour_rep_list[__].mean(axis=0)).cuda().reshape(1, self.rep_hidden_states)
                else:
                    src_hour_rep_centroid = self.src_hour_rep_list[__].mean(axis=0).reshape(1, self.rep_hidden_states)
                if isinstance(self.tgt_hour_rep_list[_], np.ndarray):
                    tgt_hour_rep_centroid = torch.from_numpy(self.tgt_hour_rep_list[_].mean(axis=0)).cuda().reshape(1, self.rep_hidden_states)
                else:
                    tgt_hour_rep_centroid = self.tgt_hour_rep_list[_].mean(axis=0).reshape(1, self.rep_hidden_states)
                dis = F.pairwise_distance(src_hour_rep_centroid, tgt_hour_rep_centroid, p=2)
                dist.append(float(dis))
            dists.append(dist)
        distance = np.array(dists)  # (self.tgt_num_space, self.src_num_spaces)
        return distance

    def match_representation(self, top_k):
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

    def calculate_match_loss(self, top_k=1):
        dist_matrix, numerator_matrix, denominator_matrix = self.match_representation(top_k)
        numerator = torch.mul(torch.from_numpy(dist_matrix), torch.from_numpy(numerator_matrix)).sum().sum()
        denominator = torch.mul(torch.from_numpy(dist_matrix), torch.from_numpy(denominator_matrix)).sum().sum()
        match_loss = np.log(numerator / denominator)
        self.match_loss = match_loss

    def domain_distance(self, rep_1, rep_2):
        if isinstance(rep_1, np.ndarray):
            rep_1 = torch.from_numpy(rep_1).cuda()
        if isinstance(rep_2, np.ndarray):
            rep_2 = torch.from_numpy(rep_2).cuda()
        return F.pairwise_distance(rep_1, rep_2, p=2)  # 2-order distance

    def calculate_da_loss(self):
        # DA loss 分子
        numerator = self.domain_distance(self.src_rep_0, self.tgt_rep_0) + self.domain_distance(self.src_rep_1,
                                                                                                self.tgt_rep_1)
        # DA loss 分母
        denominator = self.domain_distance(self.src_rep_0, self.src_rep_1) + \
                      self.domain_distance(self.src_rep_0, self.tgt_rep_1) + \
                      self.domain_distance(self.tgt_rep_0, self.src_rep_1) + \
                      self.domain_distance(self.tgt_rep_0, self.tgt_rep_1)
        da_loss = torch.div(numerator, denominator)
        # print("da_loss:", da_loss)
        self.da_loss = da_loss

    def forward(self, prob, labels):
        cls_loss = self.cls(prob, labels)
        # print('cls:', cls_loss)
        # print('match', self.gamma * self.match_loss)
        # return cls_loss + self.gamma * self.match_loss + self.da_gamma * self.da_loss
        return cls_loss + self.gamma * self.match_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

