# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
from collections import defaultdict
import json
# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

class FocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=1, epsilon=1.e-9, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha, device=device)
        else:
            self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        num_labels = input.size(-1)
        idx = target.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key[:, 0] = 0  # ignore 0 index.
        logits = torch.softmax(input, dim=-1)
        loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss = loss.sum(1)
        return loss.mean()

def train_qinggan(config, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    train_dict = defaultdict(list)
    test_dict = defaultdict(list)
    loss_fn = FocalLoss(alpha=[8000 / 1196, 8000/5660, 8000/1144])
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, qinggan_label, zhuti_label) in enumerate(train_iter):
            # print('Batch: ', i)
            outputs = model(trains)
            model.zero_grad()
            # loss = FocalLoss(alpha=[8000 / 1196, 8000/5660, 8000/1144])
            loss = loss_fn(outputs, qinggan_label)
            # print('loss: ', loss)
            loss.backward()
            # print('backward: ')
            optimizer.step()
            if total_batch % 1 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = qinggan_label.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                train_precision = metrics.precision_score(true, predic, average='macro', zero_division=0)
                train_recall = metrics.recall_score(true, predic, average='macro', zero_division=0)
                train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-5)
                dev_acc, dev_loss, dev_p, dev_r, dev_f1 = evaluate_qinggan(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*improve'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                train_dict['train_loss'].append(loss.item())   
                train_dict['train_acc'].append(train_acc)
                train_dict['train_precision'].append(train_precision)
                train_dict['train_recall'].append(train_recall)
                train_dict['train_f1'].append(train_f1)
                test_dict['dev_acc'].append(dev_acc)
                test_dict['dev_loss'].append(dev_loss.item())
                test_dict['dev_precision'].append(dev_p)
                test_dict['dev_recall'].append(dev_r)
                test_dict['dev_f1'].append(dev_f1)
                with open(config.log_path + '/train_dict.json', 'w') as json_file:
                    json.dump(train_dict, json_file, indent=4)  # indent参数使输出更加易读
                with open(config.log_path + '/test_dict.json', 'w') as json_file:
                    json.dump(test_dict, json_file, indent=4)  # indent参数使输出更加易读
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    
def evaluate_qinggan(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_fn = FocalLoss(alpha=[8000 / 1196, 8000/5660, 8000/1144])
    with torch.no_grad():
        for texts, qinggan_label, zhuti_label in data_iter:
            outputs = model(texts)
            # loss = F.cross_entropy(outputs, qinggan_label)
            loss = loss_fn(outputs, qinggan_label)
            loss_total += loss
            labels = qinggan_label.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    p = metrics.precision_score(labels_all, predict_all, average='macro', zero_division=0)
    r = metrics.recall_score(labels_all, predict_all, average='macro', zero_division=0)
    f1 =  2 * (p * r) / (p + r + 1e-5)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter), p, r, f1

def train_zhuti(config, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    alpha = 0.5
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    train_dict = defaultdict(list)
    test_dict = defaultdict(list)
    
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, qinggan_label, zhuti_label) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            # loss = F.cross_entropy(outputs, qinggan_label)
            zhuti_label = zhuti_label.float()
            # print(outputs, zhuti_label)
            loss = loss_fn(outputs, zhuti_label)
            loss.backward()
            optimizer.step()
            if total_batch % 1 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                predic = (torch.sigmoid(outputs) > alpha).float()
                # true = qinggan_label.data.cpu()
                # predic = torch.max(outputs.data, 1)[1].cpu()
                             
                train_acc = metrics.accuracy_score(predic.view(-1), zhuti_label.view(-1))
                train_precision = metrics.precision_score(predic.view(-1), zhuti_label.view(-1), average='macro', zero_division=0)
                train_recall = metrics.recall_score(predic.view(-1), zhuti_label.view(-1), average='macro', zero_division=0)
                train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-5)
                dev_acc, dev_loss, dev_p, dev_r, dev_f1 = evaluate_zhuti(config, model, dev_iter)
                                
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*improve'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                msg2 = 'Iter: {0:>6},  Train precision: {1:>5.2%},  Train recall: {2:>6.2%},  Val precision: {3:>5.2%},  Val recall: {4:>6.2%},  Time: {5} {6}'
                print(msg2.format(total_batch, train_precision, train_recall, dev_p, dev_r, time_dif, improve))
                
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("precision/train", train_precision, total_batch)
                writer.add_scalar("recall/train", train_recall, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                writer.add_scalar("precision/dev", dev_p, total_batch)
                writer.add_scalar("recall/dev", dev_r, total_batch)
                train_dict['train_loss'].append(loss.item())
                train_dict['train_acc'].append(train_acc)
                train_dict['train_precision'].append(train_precision)
                train_dict['train_recall'].append(train_recall)
                train_dict['train_f1'].append(train_f1)
                test_dict['dev_acc'].append(dev_acc)
                test_dict['dev_loss'].append(dev_loss.item())
                test_dict['dev_precision'].append(dev_p)
                test_dict['dev_recall'].append(dev_r)
                test_dict['dev_f1'].append(dev_f1)
                with open(config.log_path + '/train_dict.json', 'w') as json_file:
                    json.dump(train_dict, json_file, indent=4)  # indent参数使输出更加易读
                with open(config.log_path + '/test_dict.json', 'w') as json_file:
                    json.dump(test_dict, json_file, indent=4)  # indent参数使输出更加易读
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
def evaluate_zhuti(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    alpha = 0.5
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for texts, qinggan_label, zhuti_label in data_iter:
            outputs = model(texts)
            zhuti_label = zhuti_label.float()
            loss = loss_fn(outputs, zhuti_label)
            loss_total += loss
            labels = zhuti_label.data.cpu().numpy()
            # predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predic = (torch.sigmoid(outputs) > alpha).float().cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all.ravel(), predict_all.ravel())
    precision = metrics.precision_score(labels_all.ravel(), predict_all.ravel(), average='macro', zero_division=0)
    recall = metrics.recall_score(labels_all.ravel(), predict_all.ravel(), average='macro', zero_division=0)
    f1 =  2 * (precision * recall) / (precision + recall + 1e-5)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter), precision, recall, f1