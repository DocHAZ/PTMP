# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import roc_curve, auc
import pylab as plt
import warnings;warnings.filterwarnings('ignore')

file = open('second_label_3800.txt')
val_list = file.readlines()
lists = []
for string in val_list:
    string = string.split('\t', 1)
    lists.append(string[0])  # 只取每个string的前两项，得到的lists即为所要的列表
    a = np.array(lists)  # 将列表转化为numpy数组，
    a = a.astype(int)  # 并设定类型为intfile.close()
# print(len(a))
# print(a)

file = open('pagerank_1.txt')
val_list = file.readlines()
lists = []
for string in val_list:
    string = string.split('\t', 1)
    lists.append(string[0])  # 只取每个string的前两项，得到的lists即为所要的列表
    b = np.array(lists)  # 将列表转化为numpy数组，
    b = b.astype(float)  # 并设定类型为intfile.close()
# print(len(b))
# print(b)

fpr, tpr, thresholds = roc_curve(a, b, drop_intermediate=False)
roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# plt.show()
