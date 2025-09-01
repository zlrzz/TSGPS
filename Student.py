import pandas as pd
import numpy as np
from sklearn import linear_model
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from file_transfer import get_pair_index_exact
from data_process import length_eqaul

from model import CNN, MLP, TransformerModel, TmpModel
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import random

def Lasso_select(X_train, y_train):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train, y_train)
    lasso_coef_pair_index = np.nonzero(clf.coef_)[0]
    return clf.coef_, lasso_coef_pair_index

def get_delta(pair_index_exact, raw_gene):
    delta_in_pair_list = []
    for i in pair_index_exact:
        col1 = raw_gene[i[0]]
        col2 = raw_gene[i[1]]
        delta_in_pair = col1 - col2
        delta_in_pair_list.append(delta_in_pair)
    delta_in_pair_pandas = pd.concat(delta_in_pair_list, axis=1)
    return delta_in_pair_pandas

def combine_data(datasets):
    for name, label_name, pair_name in datasets:
        GSE = pd.read_csv('./dataset/Sepsis Data/data/%s' % name, index_col=0)
        GSE = GSE.transpose()
        label = pd.read_csv('./dataset/Sepsis Data/label/%s' % label_name, index_col=0)
        GSE, label = length_eqaul(GSE, label)

        # label = ((label > 0) * 1).values # change the label to binary classification
        pair_name = 'pair_index_exact_expressed_all.txt'

        pair = get_pair_index_exact('./iPAGE_result/Sepsis/%s' % pair_name)
        pair = pair[:50]
        delta_in_pair_pandas = get_delta(pair, GSE)

        # data = ((delta_in_pair_pandas <= 0) * (-1) + (delta_in_pair_pandas > 0) * 1).values
        data = delta_in_pair_pandas.values

        if name == datasets[0][0]:
            data_concated = data
            label_concated = label
        else:
            data_concated = np.concatenate((data_concated, data), axis=0)
            label_concated = np.concatenate((label_concated, label), axis=0)
    return data_concated, label_concated

def train_model(data_train, data_test, label_train, label_test):
    data_train = torch.from_numpy(data_train).float().to('cuda')
    data_test = torch.from_numpy(data_test).float().to('cuda')
    label_train = torch.from_numpy(label_train).long().to('cuda')
    label_test = torch.from_numpy(label_test).long().to('cuda')

    input_size = data_train.shape[1]
    hidden_size = 256
    num_classes = 3
    num_epochs = 6000
    learning_rate = 0.0001

    '''model selection'''
    model = TransformerModel(input_size, num_classes).to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=6, verbose=True,
                                                           eps=1e-10)

    loss_list = []
    acc_list = []

    old_loss = 10000
    count = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data_train)
        loss = criterion(outputs, label_train)

        if loss.item() >= old_loss:
            if loss.item() / old_loss >= 2:
                print('The difference of loss too big! {}'.format(loss.item() / old_loss))
                break
            count += 1
            if count == 19:
                print('The loss is not decreasing! {}'.format(loss.item()))
                break
        else:
            count = 0
            old_loss = loss.item()

        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

        scheduler.step(loss.item())

        if (epoch + 1) % 10 == 0:
            acc_list.append(
                100 * torch.max(model(data_train).data, 1)[1].eq(label_train).sum().item() / label_train.size(0))
            print('Epoch [{}], Loss: {:.8f}'.format(epoch + 1, loss.item()))

    model.eval()

    plt.figure(1)
    plt.plot(loss_list)
    plt.savefig('./loss.png')

    plt.figure(2)
    plt.plot(acc_list)
    plt.savefig('./acc.png')

    with torch.no_grad():
        outputs = model(data_test)
        t = outputs.cpu().numpy()[:, 1]
        t1 = outputs.cpu().numpy()[:, 2]
        t2 = np.maximum(t, t1)
        fpr, tpr, thresholds = roc_curve(label_test.cpu().numpy(), t2, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend()
        plt.savefig('./roc.png')

    with torch.no_grad():
        correct = 0
        total = 0
        outputs = model(data_test)
        _, predicted = torch.max(outputs.data, 1)
        predicted = (predicted >= 1) 
        total += label_test.size(0)
        correct += (predicted == label_test).sum().item()

    print('Test Accuracy of the model on the test: {} %'.format(100 * correct / total))

    print('The number of parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    torch.save(model.state_dict(), './model.pth')


if __name__ == '__main__':

    datasets = [('exp.gene.GSE4607.csv', 'label_GSE4607.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE8121.csv', 'label_GSE8121.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE9692.csv', 'label_GSE9692.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE9960.csv', 'label_GSE9960.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE11755.csv', 'label_GSE11755.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE13015.csv', 'label_GSE13015.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE13904.csv', 'label_GSE13904.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE25504.csv', 'label_GSE25504.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE26378.csv', 'label_GSE26378.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE26440.csv', 'label_GSE26440.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE28750.csv', 'label_GSE28750.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE32707.csv', 'label_GSE32707.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE54514.csv', 'label_GSE54514.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE57065.csv', 'label_GSE57065.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE65088.csv', 'label_GSE65088.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE65682.csv', 'label_GSE65682.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE66890.csv', 'label_GSE66890.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE69528.csv', 'label_GSE69528.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE74224.csv', 'label_GSE74224.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE95233.csv', 'label_GSE95233.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE25504.csv', 'label_GSE25504.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE65682.csv', 'label_GSE65682.csv', 'pair_index_exact_expressed_all.txt'),
                ('exp.gene.GSE66099.csv', 'label_GSE66099.csv', 'pair_index_exact_expressed_all.txt')]

    # list = random.sample(range(0, 22), 6)
    list = [3, 10, 17, 20, 15, 6]
    l1 = []
    for i in list:
        l1.append(datasets[i])
    for i in l1:
        datasets.remove(i)

    train_data, train_label = combine_data(datasets)
    test_data, test_label = combine_data(l1)

    train_data = np.array(train_data)
    train_label = np.array(train_label).reshape(-1)
    test_data = np.array(test_data)
    test_label = np.array(test_label).reshape(-1)

    count = 0
    while count < train_data.shape[0]:
        if np.isinf(train_data[count]).any():
            train_data = np.delete(train_data, count, axis=0)
            train_label = np.delete(train_label, count, axis=0)
        else:
            count += 1
    count = 0
    while count < test_data.shape[0]:
        if np.isinf(test_data[count]).any():
            test_data = np.delete(test_data, count, axis=0)
            test_label = np.delete(test_label, count, axis=0)
        else:
            count += 1

    print('Size of train data: {}  Size of test data: {}'.format(train_data.shape, test_data.shape))

    train_model(train_data, test_data, train_label, test_label)

    print('Size of train data: {}  Size of test data: {}'.format(train_data.shape, test_data.shape))
    print(list)

