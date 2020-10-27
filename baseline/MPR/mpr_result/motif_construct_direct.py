# -*-coding: utf-8 -*-
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse
from scipy.sparse import diags


def normal_matrix(a):
    # 对于非对称的矩阵，我们的归一化的方式应该是按照行来进行归一化
    for number in range(0, np.size(a, 0)):
        if number % 10000 == 0:
            print "echo is %d" % number
        row_sum = a.getrow(number).sum()
        if row_sum == 0:
            continue
        else:
            for number_col in a.indices[a.indptr[number]:a.indptr[number+1]]:
                a[number, number_col] = a[number, number_col] / row_sum
    return a


def normal_matrix_new(a):
    # 对于对称的矩阵，我们可以使用矩阵相乘的方式来进行归一化，这样的话速度比较快。
    d = a.sum(0)
    d = np.array(d)
    d = d[0]
    dd = map(lambda x: 0 if x==0 else np.power(x, -0.5), d)
    D_matrix = diags(dd, 0)
    C = D_matrix.dot(a)
    C = C.dot(D_matrix)
    a = C
    return a


def motif_times(adjacency_matrix, a, type, alpha_value):
    # 在这里，adjacency_matrix是之前我们得到的原始的邻接矩阵，可以是binary或者weighted
    # a表示的是我们需要做的是几阶的motif
    # type表示的是我们需要做的是哪一个motif

    if type == 'M1':
        for i in range(a):
            print 'echo for motif is %d' % i
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            U_matrix = adjacency_matrix - B_matrix_spare
            U_matrix = np.abs(U_matrix)
            U_tran = np.transpose(U_matrix)
            result_1 = U_matrix.dot(U_matrix)
            result_1 = result_1.multiply(U_tran)
            adjacency_matrix = result_1
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            adjacency_tran = np.transpose(adjacency_matrix)
            adjacency_matrix = adjacency_matrix + adjacency_tran
            if len(adjacency_matrix.data) > 0:
                print np.max(adjacency_matrix.data), adjacency_matrix.nnz
    elif type == 'M2':
        for i in range(a):
            print 'echo for motif is %d' % i
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            U_matrix = adjacency_matrix - B_matrix_spare
            U_matrix = np.abs(U_matrix)
            U_tran = np.transpose(U_matrix)
            result_1 = B_matrix_spare.dot(U_matrix)
            result_1 = result_1.multiply(U_tran)
            result_2 = U_matrix.dot(B_matrix_spare)
            result_2 = result_2.multiply(U_tran)
            result_3 = U_matrix.dot(U_matrix)
            result_3 = result_3.multiply(B_matrix_spare)
            adjacency_matrix = result_1 + result_2
            adjacency_matrix = adjacency_matrix + result_3
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            adjacency_tran = np.transpose(adjacency_matrix)
            adjacency_matrix = adjacency_matrix + adjacency_tran
            if len(adjacency_matrix.data) > 0:
                print np.max(adjacency_matrix.data), adjacency_matrix.nnz
    elif type == 'M3':
        for i in range(a):
            print 'echo for motif is %d' % i
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            U_matrix = adjacency_matrix - B_matrix_spare
            U_matrix = np.abs(U_matrix)
            result_1 = B_matrix_spare.dot(B_matrix_spare)
            result_1 = result_1.multiply(U_matrix)
            result_2 = B_matrix_spare.dot(U_matrix)
            result_2 = result_2.multiply(B_matrix_spare)
            result_3 = U_matrix.dot(B_matrix_spare)
            result_3 = result_3.multiply(B_matrix_spare)
            adjacency_matrix = result_1 + result_2
            adjacency_matrix = adjacency_matrix + result_3
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            adjacency_tran = np.transpose(adjacency_matrix)
            adjacency_matrix = adjacency_matrix + adjacency_tran
            if len(adjacency_matrix.data) > 0:
                print np.max(adjacency_matrix.data), adjacency_matrix.nnz
    elif type == 'M4':
        for i in range(a):
            print 'echo for motif is %d' % i
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            result_1 = B_matrix_spare.dot(B_matrix_spare)
            result_1 = result_1.multiply(B_matrix_spare)
            adjacency_matrix = result_1
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            if len(adjacency_matrix.data) > 0:
                print np.max(adjacency_matrix.data), adjacency_matrix.nnz
    elif type == 'M5':
        for i in range(a):
            print 'echo for motif is %d' % i
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            U_matrix = adjacency_matrix - B_matrix_spare
            U_matrix = np.abs(U_matrix)
            U_tran = np.transpose(U_matrix)
            result_1 = U_matrix.dot(U_matrix)
            result_1 = result_1.multiply(U_matrix)
            result_2 = U_matrix.dot(U_tran)
            result_2 = result_2.multiply(U_matrix)
            result_3 = U_tran.dot(U_matrix)
            result_3 = result_3.multiply(U_matrix)
            adjacency_matrix = result_1 + result_2
            adjacency_matrix = adjacency_matrix + result_3
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            adjacency_tran = np.transpose(adjacency_matrix)
            adjacency_matrix = adjacency_matrix + adjacency_tran
            if len(adjacency_matrix.data) > 0:
                print np.max(adjacency_matrix.data), adjacency_matrix.nnz
    elif type == 'M6':
        for i in range(a):
            print 'echo for motif is %d' % i
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            U_matrix = adjacency_matrix - B_matrix_spare
            U_matrix = np.abs(U_matrix)
            U_tran = np.transpose(U_matrix)
            result_1 = U_matrix.dot(B_matrix_spare)
            result_1 = result_1.multiply(U_matrix)
            result_2 = B_matrix_spare.dot(U_tran)
            result_2 = result_2.multiply(U_tran)
            result_3 = U_tran.dot(U_matrix)
            result_3 = result_3.multiply(B_matrix_spare)
            adjacency_matrix = result_1 + result_2
            adjacency_matrix = adjacency_matrix + result_3
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            if len(adjacency_matrix.data) > 0:
                print np.max(adjacency_matrix.data), adjacency_matrix.nnz
    elif type == 'M7':
        for i in range(a):
            print 'echo for motif is %d' % i
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            U_matrix = adjacency_matrix - B_matrix_spare
            U_matrix = np.abs(U_matrix)
            U_tran = np.transpose(U_matrix)
            result_1 = U_tran.dot(B_matrix_spare)
            result_1 = result_1.multiply(U_tran)
            result_2 = B_matrix_spare.dot(U_matrix)
            result_2 = result_2.multiply(U_matrix)
            result_3 = U_matrix.dot(U_tran)
            result_3 = result_3.multiply(B_matrix_spare)
            adjacency_matrix = result_1 + result_2
            adjacency_matrix = adjacency_matrix + result_3
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            if len(adjacency_matrix.data) > 0:
                print np.max(adjacency_matrix.data), adjacency_matrix.nnz
    return adjacency_matrix


def construct_motif(txt_name, times, type, alpha_value):
    # 先从txt中读取数据，txt中每一行的格式为a,b。表示的是a引用了b
    entry = []
    f = open(txt_name)
    while True:
        line = f.readline()
        if line:
            temp = []
            line = line.replace('\n', '')
            line = line.split(' ')
            for i in range(len(line)):
                temp.append(int(line[i]))
            entry.append(temp)
        else:
            break
    spare_array_row = []
    spare_array_col = []
    spare_array_data = []
    entry_all = []
    for i in range(len(entry)):
        spare_array_row.append(entry[i][0])
        spare_array_col.append(entry[i][1])
        spare_array_data.append(1)
        entry_all.append(entry[i][0])
        entry_all.append(entry[i][1])
    entry_unique = np.unique(entry_all)
    # entry_unique中包含了所有的author的id
    newspare_array_row = []
    newspare_array_col = []
    counttt = 0
    # 为了要构建矩阵，需要将author的id进行连续性的编码
    for nnn in range(len(spare_array_row)):
        if counttt % 300000 == 0:
            print 'echo is %d' % counttt
        counttt += 1
        id1 = spare_array_row[nnn]
        id2 = spare_array_col[nnn]
        id1new = np.where(entry_unique == id1)[0][0]
        id2new = np.where(entry_unique == id2)[0][0]
        newspare_array_row.append(id1new)
        newspare_array_col.append(id2new)
    maxnum = len(entry_unique)
    adjacency_matrix = csr_matrix((spare_array_data, (newspare_array_row, newspare_array_col)), shape=(maxnum, maxnum), dtype = np.float64)
    data_array = adjacency_matrix.data
    lennn = data_array.shape[0]
    # 下面这一句adjacency_matrix.data主要是为了能够构建binary情况下的矩阵，如果需要weighted的矩阵，必须注释这句话
    adjacency_matrix.data = np.ones((1, lennn), dtype=np.float64)[0]
    # print adjacency_matrix.nnz
    result_B = adjacency_matrix.copy()
    result_B = normal_matrix(result_B)
    # result_B表示的是经过归一化之后的邻接矩阵
    result_C = motif_times(adjacency_matrix, times, type, alpha_value)
    # result_C表示的是经过归一化的motif矩阵
    # 下面使用参数alpha_value对两个矩阵进行融合
    # alpha_value = 0.1
    result_temp1 = result_B.multiply(alpha_value).tolil()
    result_temp2 = result_C.multiply(1-alpha_value).tolil()
    result_D = result_temp1 + result_temp2
    result_D = result_D.tocsr()
    print maxnum
    return result_D, entry_unique


def binary(a_original):
    # 这个函数是将矩阵进行binary的
    a = a_original.copy()
    for number in range(0, np.size(a, 0)):
        if number % 10000 == 0:
            print "echo is %d" % number
        for number_col in a.indices[a.indptr[number]:a.indptr[number+1]]:
            a[number, number_col] = 1
    return a


if __name__ == '__main__':
    result_D, entry_unique = construct_motif('author.txt', 1, 'M6', 0.8)
