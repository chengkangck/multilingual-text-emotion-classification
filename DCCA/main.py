# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from DeepCCAModels import DeepCCA
from utils import load_data, classify
import time
import logging
from classification_utilities import display_cm
import csv
import pandas as pd
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import torch.nn as nn
torch.set_default_tensor_type(torch.DoubleTensor)
def count_chinese(data):
    a1 = 0
    b1 = 0
    c1 = 0
    d1 = 0
    e1 = 0
    f1 = 0
    g1 = 0
    n1 = 0
    for i in data:
        n1 = n1 + 1
        if i == 0:
            a1 = a1 + 1
        if i == 1:
            b1 = b1 + 1
        if i == 2:
            c1 = c1 + 1
        if i == 3:
            d1 = d1 + 1
        if i == 4:
            e1 = e1 + 1
        if i == 5:
            f1 = f1 + 1
        if i == 6:
            g1 = g1 + 1
    print('情感happiness有:', a1)
    print('情感sadness有:', b1)
    print('情感disgust有:', c1)
    print('情感anger有:', d1)
    print('情感fear有:', e1)
    print('情感surprise有:', f1)
    print('情感like有:', g1)
    print('共有数据',n1)
    print('...........................')
# 定义训练时网络的输入输出尺寸模型
class Solver():
    def __init__(self, model, linear_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=torch.device('cpu')):
        self.model = nn.DataParallel(model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.device = device

        self.linear_cca = linear_cca

        self.outdim_size = outdim_size

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("DCCA.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)
# 定义训练函数
    def fit(self, x1, x2, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint='checkpoint.model'):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        x1.to(self.device)
        x2.to(self.device)

        data_size = x1.size(0)

        if vx1 is not None and vx2 is not None:
            best_val_loss = 0
            vx1.to(self.device)
            vx2.to(self.device)
        if tx1 is not None and tx2 is not None:
            tx1.to(self.device)
            tx2.to(self.device)

        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x1 = x1[batch_idx, :]
                # batch_x2 = x2[batch_idx, :]
                batch_x2 = x1[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                loss = self.loss(o1, o2)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.test(vx1, vx2)
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))
            else:
                torch.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss))
        # 使用DCCA模型
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(x1, x2)
            self.train_linear_cca(outputs[0], outputs[1])

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        if vx1 is not None and vx2 is not None:
            loss = self.test(vx1, vx2)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

        if tx1 is not None and tx2 is not None:
            loss = self.test(tx1, tx2)
            self.logger.info('loss on test data: {:.4f}'.format(loss))

# 定义训练DCCA的测试数据
    def test(self, x1, x2, use_linear_cca=False):
        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)

            if use_linear_cca:
                print("Linear CCA started!")
                outputs = self.linear_cca.test(outputs[0], outputs[1])
                return np.mean(losses), outputs
            else:
                return np.mean(losses)

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                # batch_x2 = x2[batch_idx, :]
                batch_x2 = x1[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        return losses, outputs


if __name__ == '__main__':
    ############
    # 使用GPU
    device = torch.device('cpu')
    print("Using", torch.cuda.device_count(), "GPUs")


    # 输出向量的维度
    outdim_size = 100

    # 两个输入向量的维度
    input_shape1 = 400
    input_shape2 = 400

    # 每层中有节点的层数
    layer_sizes1 = [1024, 1024, 1024, outdim_size]
    layer_sizes2 = [1024, 1024, 1024, outdim_size]

    # 训练网络所需参数
    learning_rate = 1e-3
    epoch_num = 3
    batch_size = 800

    # 正则化参数
    reg_par = 1e-5

    # 仅使用顶部奇异值来计算相关性
    use_all_singular_values = False

    # 将线性CCA应用于从网络中提取的学习特征
    apply_linear_cca = True

# 开始使用模型
    model = DeepCCA(layer_sizes1, layer_sizes2, input_shape1,
                    input_shape2, outdim_size, use_all_singular_values, device=device).double()
    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()
#奇异值分解
    solver = Solver(model, l_cca, outdim_size, epoch_num, batch_size,
                    learning_rate, reg_par, device=device)
# 加载中文数据集
    firl = './chinese.csv'
    train11 = np.load('./chinese.npy')
    train11 = train11.astype(float)
    train1 = torch.from_numpy(train11)
# 加载英文数据集
    fir1 = './english.csv'
    train22 = np.load('./english.npy')
    train22 = train22.astype(float)
    train2 = torch.from_numpy(train22)
# 进行奇异值分解
    solver.fit(train1, train2, train1, train2, train1, train2)
# 设置数据集尺寸，使其输入进网络模型
    set_size = [0, train1.size(0), train1.size(
        0) + train1.size(0), train1.size(0) + train1.size(0) + train1.size(0)]
# 以损失函数来决定输出
    loss, outputs = solver.test(torch.cat([train1, train1, train1], dim=0), torch.cat(
        [train2, train2, train2], dim=0), apply_linear_cca)
# 得到新数据集
    new_data = []
    print('outputs',outputs)
    data=outputs[0].tolist()
    data_1 = []
    n = 0
# 列出新数据集
    for one_line in data:
        data_1.append(one_line)
        n = n + 1

    labels1 = ['happiness', 'sadness', 'disgust', 'anger', 'fear', 'surprise', 'like',  ]

    [y1,y,y_p]=classify(firl)
    print('训练集：')
    count_chinese(y1)
    print('测试集：')
    count_chinese(y)
    # 得出混淆矩阵
    cv_conf = confusion_matrix(y, y_p)
    display_cm(cv_conf, labels1, display_metrics=True, hide_zeros=True)
    micro_f1 = f1_score(y, y_p, average='micro')
    macro_f1 = f1_score(y, y_p, average='macro')
    print('macro_f1', macro_f1)
    print('micro_f1', micro_f1)


