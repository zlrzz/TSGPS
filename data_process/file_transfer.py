import csv

import numpy as np
import pandas as pd
import openpyxl as px


def read_file(file_name):
    with open(file_name, 'r') as file:
        return file.read()


'''Write kegg symbols to csv file'''


def write_to_csv(file_name):
    file = read_file(file_name)
    t_array = np.array(file.split('\n'))
    result = []
    for i in range(len(t_array)):
        t = t_array[i].split('\t')
        result.append(t[2:])
    result.remove([])
    pd.DataFrame(result).to_csv('c2.cp.kegg.v7.2.symbols.csv', index=False, header=True, sep='\t')


'''Write labels to csv file'''


def xlsx_to_csv(file_name):
    wb = px.load_workbook(file_name + '.xlsx')
    sheet = wb.active
    data = sheet.values
    data = list(data)
    data = np.array(data, dtype=str)
    pd.DataFrame(data).to_csv('%s.csv' % file_name, index=False, header=False)


'''Get pair_index from the stored txt file'''


def get_pair_index_exact(file_name):
    pair = read_file(file_name)
    pair = pair.replace('(', '').replace(')', '').replace(' ', '').replace('[', '').replace(']', '').replace('\'',
                                                                                                             '').split(
        ',')
    new_pair = []
    k = 0
    while k < len(pair):
        new_pair.append((pair[k], pair[k + 1]))
        k += 2
    new_pair = np.array(new_pair)
    return new_pair


'''Get spesis data from txt file to csv with ILMN_Gene'''


def get_sepsis_data(file_name1, file_name2):
    file = read_file(file_name2)
    t_array = np.array(file.split('\n'))
    compare = {}
    gene_col = []
    for i in range(len(t_array)):
        t = t_array[i].split(',')
        if len(t) == 1:
            continue
        compare.update({t[0]: t[1]})
        if t[1] not in gene_col:
            gene_col.append(t[1])

    file = read_file(file_name1)
    file = file.replace('"', '')
    t_array = np.array(file.split('\n'))
    result_t = []
    for i in range(len(t_array)):
        t = t_array[i].split('\t')
        if t[0] in compare:
            t[0] = compare[t[0]]
        result_t.append(t)
    result = [result_t[0]]
    for i in range(len(gene_col)):
        count = 0
        t = np.zeros(len(result_t[0]) - 1, dtype=float)
        for j in range(len(result_t)):
            if result_t[j][0] == gene_col[i]:
                count += 1
                if result_t[j][1] == '':
                    continue
                t = (t + np.array(result_t[j][1:], dtype=float)) / count
        if t[1] != 0:
            result.append([gene_col[i]] + list(t))
    with open('./dataset/Sepsis Data/data/exp.gene.GSE9960.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)


'''Get the GPL data'''


def get_GPL(file_name):
    file = read_file(file_name)
    t_array = np.array(file.split('\n'))
    result = []
    for i in range(len(t_array)):
        t = t_array[i].split(',')
        id = t[0]
        if len(t) == 1:
            continue
        symbols = t[1]
        if symbols == 'na':
            continue
        if len(symbols) > 1:
            symbols = symbols.split('///')
            if len(symbols) > 1:
                symbol = symbols[0]
            else:
                symbol = symbols[0]
            result.append([id, symbol])
    pd.DataFrame(result).to_csv('GPL13667.csv', index=False, header=False)




if __name__ == '__main__':
    # get_sepsis_data('./dataset/Sepsis Data/GSE9960_series_matrix.txt', './dataset/Sepsis Data/GPL570.csv')
    xlsx_to_csv('./dataset/Sepsis Data/label/label_GSE4607')
    # xlsx_to_csv('./dataset/Sepsis Data/GPL13667')
    # get_GPL('./dataset/Sepsis Data/GPL13667.csv')
