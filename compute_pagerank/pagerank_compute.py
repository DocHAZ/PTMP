#!/usr/bin/env python
# -*-coding: utf-8 -*-

import operator
import numpy as np
import pandas as pd


class compute_pagerank(object):
    def __init__(self,file1,file2):
        self.file1=file1
        self.file2=file2


    def time_edge(self,file):
        event_list=[]
        for i in file:
            file3 = i.rstrip('\n')
            event_list.append(file3)
        return event_list


    def bl_list(self,event):
        event_list=self.time_edge(event)
        b_list = []
        for i in event_list:
            m = i.split(' ')
            b_list.append(m)
        return b_list

    def travel_ori_event(self):
        ori_event=self.bl_list(self.time_edge(self.file2))

        # 按照时间戳，从初始节点开始遍历原始数据每一个时间窗口内的时间边
        timestamp = []
        s = 14 #原数据排序后的第一个时间戳
        t = 10 #时间戳
        compare = []
        for q in range(1,3):
            for i in ori_event:
                e = int(i[2])
                f = int(i[0])
                g = int(i[1])
                i[2] = e
                i[0] = f
                i[1] = g
                if s+(q-1)*t<=i[2]<s+q*t:
                    compare.append(i)
            timestamp = compare
        return timestamp

    def construct_dict(self):
        # 将时间窗口内的原始数据中的时间边与motif group中的时间边进行比较
        evs = []
        for i in self.bl_list(self.file1):
            evs.append(i)
        key=[]
        value=[]
        for i in self.travel_ori_event():
                js = 0
                for q in evs:
                    if operator.eq(str(i).replace("'",''),str(q).replace("'",'')) is True:
                        js = js+1
                    else:
                        pass
                key.append(str(i).replace("'",''))
                value.append(js)
        my=dict(zip(key,value))
        return my

    def dict_change(self):
        Myset = set()
        change_key = []
        change_value = []
        for k, v in self.construct_dict().items():
            values = 0
            key_pro = str(str(k).strip('[').split(',')[:-1]).replace("'", '')


            if key_pro not in Myset:
                Myset.add(key_pro)
                change_key.append(key_pro)
                for k2, v2 in self.construct_dict().items():
                    key_pro2 = str(str(k2).strip('[').split(',')[:-1]).replace("'", '')
                    if key_pro == key_pro2:
                        values = values + int(self.construct_dict().get(k2))
                change_value.append(values)
            else:
                continue
        change_Mydict = dict(zip(change_key, change_value))
        return change_Mydict

    def dict_list(self):

        sample_key_list = []
        for item in self.dict_change().keys():
            sample_key_list.append(item)


        sample_value_list = []
        for item in self.dict_change().values():
            sample_value_list.append(item)
        return sample_key_list, sample_value_list

    def construct_newdict(self):
        sample_key_list,sample_value_list=self.dict_list()
        values=[]
        for i in sample_value_list:
            # print(i)
            a=int(i)
            values.append(a)


        new_list = []
        for k in sample_key_list:
            k = eval(k)
            list_k = []
            list_k.append(k[0])
            list_k.append(k[1])
            new_list.append(list_k)
        return new_list

    def list_max(self):
        new_id = []
        for i in self.construct_newdict():
            nn = []
            for a in i:
                t = int(a)
                nn.append(t)
            new_id.append(nn)
        # 定义一个新的列表，通过上一个列表中的键，找出键中数字的最大值，并将其赋值给矩阵的维数
        maxx = []
        for i in new_id:
            a = max(i)
            maxx.append(a)
        d = max(maxx)
        return d,new_id

    def matrix_compute(self):
        d, new_id=self.list_max()
        wx,values=self.dict_list()
        motif_weighted_matrix = np.zeros((d,d))
        for it in range(len(new_id)):
            motif_weighted_matrix[new_id[it][0]-1][new_id[it][1]-1]=values[it]
        print(motif_weighted_matrix)

        # 计算motif出度矩阵
        a = np.array(motif_weighted_matrix)
        b1 = pd.DataFrame(a)
        b2 = b1.sum(axis=1)
        motif_degree_matrix = np.zeros((d,d))
        row, col = np.diag_indices_from(motif_degree_matrix)
        motif_degree_matrix[row,col] = np.array(b2)
        print(motif_degree_matrix)
        return motif_degree_matrix,motif_weighted_matrix


    def pagerank_compute(self):
        #根据Motif出度矩阵和加权motif邻接矩阵计算pagerank值
        motif_degree_matrix, motif_weighted_matrix=self.matrix_compute()
        A = np.mat(motif_degree_matrix)
        B = np.mat(motif_weighted_matrix)

        alpha_value = 0.8
        # 线性
        result_1 = np.multiply(alpha_value,A)
        result_2 = np.multiply((1 - alpha_value),B)
        result_3 = result_1 + result_2
        print(result_3)

        # 归一化
        a = np.array(result_3)
        b1 = pd.DataFrame(a)
        b2=b1.sum(axis=1)
        b = b1.div(b2, axis='rows').fillna(0)
        b=b.T
        d = 0.8
        # 对于一个大图，n是矩阵的维数
        n = 4
        c = np.mat(np.identity(n))
        e = c-d*b
        f = (1-d)/n
        g = np.ones((1,4))*f
        g=g.T
        # 对矩阵求逆
        h = np.linalg.inv(e)
        # 计算Pagerank值
        Pagerank_result = np.dot(h,g)
        list1 = list(range(1,5))
        print(Pagerank_result.shape)#查看矩阵维数

        nvDict = dict(zip(list1,Pagerank_result.tolist()))#将Numpy matrix类型的数据转换为列表类型
        print(nvDict)
        return nvDict

    def write_file(self):
        # 写文件
        nvDict=self.pagerank_compute()
        with open("tmp_value.txt", 'w') as outfile:
           for k,v in nvDict.items():
               outfile.writelines(str(k) +" "+ str(v).strip('[').strip(']')+'\n')

if __name__ == '__main__':

    with open('tuple.txt','r') as f:
        file1 = f.readlines()
        f.close()
    with open('input_graph.txt','r') as f:
        file2 = f.readlines()
        f.close()

    TMP=compute_pagerank(file1,file2)
    TMP.write_file()






