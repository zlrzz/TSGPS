import pandas as pd
import numpy as np
import time
from queue import PriorityQueue

'''Factoria function by method of merge'''


def factoria(N):
    N2 = N
    a = list(range(1, N + 1))
    while N2 > 1:
        N1 = N2 % 2
        N2 = N2 // 2 + N1
        for i in range(N2 - N1):
            a[i] *= a[N2 + i]
        a = a[0:N2]
    return a[0]


'''Calculate the combination for fisher exact test'''


def comb_mine(N, k):
    if N == k or N == 0 or k == 0:
        return 1
    return factoria(N) // (factoria(k) * factoria(N - k))


'''Check the length and data value of RNA sequence and label sequence'''


def length_eqaul(GSE, GSE_label):
    GSE_len = GSE.shape[0]
    label_len = GSE_label.shape[0]
    while GSE_len != label_len:
        if GSE_len != label_len:
            count_drop = 0
            if GSE_len > label_len:
                for i in range(GSE_len):
                    i = i - count_drop
                    if GSE.index[i] not in GSE_label.index:
                        GSE = GSE.drop(index=GSE.index[i])
                        count_drop += 1
            else:
                for i in range(label_len):
                    i = i - count_drop
                    if GSE_label.index[i] not in GSE.index:
                        GSE_label = GSE_label.drop(index=GSE_label.index[i])
                        count_drop += 1
        GSE_len = GSE.shape[0]
        label_len = GSE_label.shape[0]
    return GSE, GSE_label


'''Combine all data and label'''


def combine_all_data(datasets):
    GSE_final = pd.DataFrame()
    label_final = pd.DataFrame()
    for name, label, _ in datasets:
        GSE = pd.read_csv('./dataset/ttttttttt/Staphy Data/data/%s' % name, index_col=0)
        GSE_label = pd.read_csv('./dataset/ttttttttt/Staphy Data/label/%s' % label, index_col=0)
        GSE = GSE.transpose()
        GSE, GSE_label = length_eqaul(GSE, GSE_label)
        GSE = GSE.transpose()
        GSE_label = GSE_label.transpose()
        if GSE_final.empty:
            GSE_final = GSE
            label_final = GSE_label
            continue
        GSE_final = pd.concat([GSE_final, GSE], join='inner', axis=1)
        label_final = pd.concat([label_final, GSE_label], axis=1)
        print(name)
        print(GSE_final.shape, label_final.shape)
    GSE_final = GSE_final.transpose()
    label_final = label_final.transpose()
    GSE_final.to_csv('./dataset/ttttttttt/Staphy Data/data.csv')
    label_final.to_csv('./dataset/ttttttttt/Staphy Data/label.csv')
    print(GSE_final.shape, label_final.shape)


'''drop useless pair'''


def drop(storege, GSE):
    symbols = pd.read_csv(storege, sep='\t')
    symbols = symbols.values
    gene_label = GSE.columns
    for i in range(len(gene_label)):
        if gene_label[i] not in symbols:
            GSE = GSE.drop(columns=gene_label[i])
    return GSE


'''iPage and Fisher exact test'''


def fisher_exact_test(GSE, GSE_label):
    data = GSE.values
    label = GSE_label.values
    columns = GSE.columns
    len_gene = len(columns)
    pair_index_exact = []
    PQ = PriorityQueue()
    for i in range(len_gene - 1):
        if i % 100 == 0:
            print('=========RNAseq %d=============' % i)
            print("--- %s seconds ---" % (time.time() - start_time))
        for j in range(i + 1, len_gene):
            pair_each = np.array(data[:, i]) > np.array(data[:, j])
            a = np.sum((pair_each > 0.5) * (np.array(label < 0.5)).squeeze(1))
            c = np.sum((pair_each > 0.5) * (np.array(label > 0.5)).squeeze(1))
            b = np.sum((pair_each < 0.5) * (np.array(label < 0.5)).squeeze(1))
            d = np.sum((pair_each < 0.5) * (np.array(label > 0.5)).squeeze(1))
            n = a + b + c + d
            p = comb_mine(a + b, a) * comb_mine(c + d, c) / comb_mine(n, a + c)
            PQ.put((p, columns[i], columns[j]))
    for i in range(PQ.qsize()):
        t = PQ.get()[1:3]
        pair_index_exact.append(t)
    return pair_index_exact


if __name__ == '__main__':
    datasets = [('GSE11907.csv', 'label_GSE11907.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE13015.csv', 'label_GSE13015.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE16129-GPL6106.csv', 'label_GSE16129-GPL6106.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE16129_GPL96.csv', 'label_GSE16129-GPL96.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE22098.csv', 'label_GSE22098.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE25504_GPL6947.csv', 'label_GSE25504-GPL6947.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE30119.csv', 'label_GSE30119.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE33341-GPL571.csv', 'label_GSE33341-GPL571.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE69528.csv', 'label_GSE69528.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE100165.csv', 'label_GSE100165.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE6269-GPL96.csv', 'label_GSE6269-GPL96.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE6269_GPL570.csv', 'label_GSE6269-GPL570.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE13015-GPL6106.csv', 'label_GSE13015_GPL6106.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE23140.csv', 'label_GSE23140.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE25504-GPL13667.csv', 'label_GSE25504-GPL13667.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE47172.csv', 'label_GSE47172.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE68004.csv', 'label_GSE68004.csv', 'pair_index_exact_expressed_all.txt'),
                ('GSE69528.csv', 'label_GSE69528.csv', 'pair_index_exact_expressed_all.txt')]

    combine_all_data(datasets)