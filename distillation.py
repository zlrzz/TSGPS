import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from data_process import length_eqaul
from file_transfer import get_pair_index_exact
from model import MLP, TmpModel, TransformerModel
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

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
        GSE = pd.read_csv('./dataset/RSV Data/%s' % name, index_col=0)
        # GSE = GSE.transpose()
        GSE = GSE.apply(np.log)
        label = pd.read_csv('./dataset/RSV Data/%s' % label_name, index_col=0)
        GSE, label = length_eqaul(GSE, label)

        # label = ((label > 0) * 1).values # change the label to binary classification
        pair_all = get_pair_index_exact('./iPAGE_result/RSV/%s' % pair_name)
        pair = pair_all[:35]
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


def distillation_learning(data_train, data_test, label_train, label_test):
    data_train = torch.from_numpy(data_train).float().to('cuda')
    data_test = torch.from_numpy(data_test).float().to('cuda')
    label_train = torch.from_numpy(label_train).long().to('cuda')
    label_test = torch.from_numpy(label_test).long().to('cuda')

    input_size = data_train.shape[1]
    num_classes_student = 3
    num_classes_teacher = 3
    epochs = 6000
    learning_rate = 0.0001
    T = 2
    soft_target_loss_weight = 0.1
    ce_loss_weight = 0.9
    loss_list = []

    old_loss = 10000
    count = 0

    teacher = TmpModel(input_size, num_classes_teacher).to('cuda')
    student = TransformerModel(input_size, num_classes_student).to('cuda')

    teacher.load_state_dict(torch.load('./d_model.pth'))

    optimizer = optim.AdamW(student.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=6, verbose=True,
                                                           eps=1e-10)
    
    teacher.eval()
    student.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher(data_train)
        student_logits = student(data_train)
        soft_targets = F.softmax(teacher_logits / T, dim=-1)
        soft_prob = F.log_softmax(student_logits / T, dim=-1)
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T ** 2)
        label_loss = criterion(student_logits, label_train)
        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
    
        if loss.item() >= old_loss:
            if loss.item() / old_loss >= 2:
                print('The difference of loss too big! {}'.format(loss.item() / old_loss))
                break
            count += 1
            # print('The difference of loss: {}'.format(loss.item() / old_loss))
            if count == 19:
                print('The loss is not decreasing! {}'.format(loss.item()))
                break
        else:
            count = 0
            old_loss = loss.item()
    
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
    
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}], Loss: {:.8f}'.format(epoch + 1, loss.item()))
    
    plt.figure(1)
    plt.plot(loss_list)
    plt.savefig('./d_loss.png')

    teacher.eval()
    student.eval()

    with torch.no_grad():
        outputs = student(data_test)
        t = outputs.cpu().numpy()[:, 1]
        t1 = outputs.cpu().numpy()[:, 2]
        t2 = np.maximum(t, t1)
        fpr, tpr, thresholds = roc_curve(label_test.cpu().numpy(), t2, pos_label=2)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('RSV ROC')
        plt.legend()
        # plt.savefig('./d_roc.png')

    with torch.no_grad():
        correct = 0
        total = 0
        student_logits = student(data_test)
        _, predicted = torch.max(student_logits, 1)
        predicted = (predicted >= 1) * 2
        total += label_test.size(0)
        correct += (predicted == label_test).sum().item()
        print('Accuracy: {}%'.format(100 * correct / total))
        print('The number of parameters: {}'.format(sum(p.numel() for p in student.parameters())))
        torch.save(student.state_dict(), './d_1_model.pth')


if __name__ == '__main__':

    datasets = [('data.csv', 'label.csv', 'pair_index_exact_expressed_all.txt')]

    data, label = combine_data(datasets)

    data = np.array(data)
    label = np.array(label).reshape(-1)

    count = 0
    while count < data.shape[0]:
        if np.isinf(data[count]).any() or np.isnan(data[count]).any():
            data = np.delete(data, count, axis=0)
            label = np.delete(label, count, axis=0)
        else:
            count += 1

    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=22)

    print('Size of train data: {}  Size of test data: {}'.format(train_data.shape, test_data.shape))

    distillation_learning(train_data, test_data, train_label, test_label)

    print('Size of train data: {}  Size of test data: {}'.format(train_data.shape, test_data.shape))
