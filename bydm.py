#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import numpy as np


# In[ ]:


data = pd.read_csv('E:/睡眠中的人体压力检测数据集.csv')
data.head(10)


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.duplicated().sum()


# In[ ]:


plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
plt.figure(figsize=(6,4))#图片大小
sns.countplot(x='sl',data=data)#绘图 选择字段以及数据
plt.title('不同压力水平样本数量',fontsize=18)#图片名称
plt.xticks(fontsize=15)#x轴刻度值字体大小
plt.yticks(fontsize=15)
plt.xlabel('x',fontsize=15)#x轴标签
plt.ylabel('y',fontsize=15)


# In[ ]:


num_field = [
'sl',
'sr',
'rr',
't',
'lm',
'bo',
'rem',
'sr.1',
'hr'
]#选取指标
f,ax = plt.subplots(figsize=(10, 8))#设置图片大小
sns.heatmap(data[num_field].corr(),annot=True,
                    cmap = sns.diverging_palette(220, 10, as_cmap = True),
                    linewidths=.9, fmt= '.3f',ax = ax)#画图
plt.title('睡眠特征相关系数',fontsize=20)#图片题目


# In[ ]:


# one-hot
pd.get_dummies(data=data, columns=['t', 'bo', 'sr.1'], prefix_sep="_")


# In[ ]:


# 将需要进行One-Hot编码的列转换为category类型
data_trans = data.astype('category')
# 进行One-Hot编码
pd.get_dummies(data_trans)


# In[ ]:


# 哑编码
pd.get_dummies(data=data, columns=['t', 'bo', 'sr.1'], prefix_sep="-", drop_first=True)


# In[ ]:


# 分离特征和标签
X = data.drop('sl',axis=1)
y = data['sl']

# 划分
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=10,stratify=y)

# 查看训练集测试集类别分布
print('训练集维度：',X_train.shape)
print('测试集维度：',X_test.shape)
print('-----------------------')
print('训练集类别分布：',round(y_train.value_counts()[1]/len(y_train),2))
print('测试集类别分布：',round(y_test.value_counts()[1]/len(y_test),2))


# In[ ]:


# 模型构建与拟合
lr = LogisticRegression(random_state=10, max_iter=1000) # 增加max_iter参数
lr.fit(X_train, y_train)


# In[ ]:


## 模型预测标签
y_pred = lr.predict(X_test)
## 模型预测概率
lr.classes_


# In[ ]:


lr.predict_proba(X_test)


# In[ ]:


y_prob_1 = lr.predict_proba(X_test)[:,1]
y_prob_0 = lr.predict_proba(X_test)[:,0]


# In[ ]:


# 模型评价（内置方法）——分类正确率
round(lr.score(X_test,y_test),2)


# In[ ]:


# 模型评价（Sklearn函数）——分类正确率
# 用accuracy_score函数计算y_test和y_pred 的准确率。将准确率保留两位小数，使用round函数进行四舍五入。
round(accuracy_score(y_true=y_test,y_pred=y_pred),2)


# In[ ]:


print(classification_report(y_pred=y_pred,y_true=y_test))


# In[ ]:


#生成一个混淆矩阵conf_matrix
conf_matrix = confusion_matrix(y_pred=y_pred,y_true=y_test,labels=[0,1])

#返回一个形状为(2,2)的二维数组，表示模型在测试集上的混淆矩阵
conf_matrix


# In[ ]:


# 热力图
fig,ax = plt.subplots(figsize=(5, 4))
sns.heatmap(conf_matrix, ax=ax, 
            annot=True, # 在每个方格显示数字
            annot_kws={'size':15},
            fmt='d'# 定义方格中字体的大小、颜色、样式
           )
ax.set_ylabel('真实标签')
ax.set_xlabel('预测标签')
ax.set_title('混淆矩阵热力图')
plt.show()


# In[ ]:


#生成一个ROC曲线所需要的false positive rate（FPR）和true positive rate（TPR），以及相应的阈值threshold
fpr,tpr,threshold = roc_curve(y_score=y_prob_1,y_true=y_test,pos_label=1)
# auc值
auc(fpr,tpr)


# In[ ]:


# ROC曲线
plt.figure(figsize=(8,8), facecolor='white')
plt.plot(fpr, tpr, color='red')

# 绘制虚线
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)
plt.title('ROC曲线')


# In[ ]:


#返回三个数组，分别为精确率（precision）、召回率（recall）和阈值threshold。
precision,recall,threshold = precision_recall_curve(y_true=y_test,probas_pred=y_prob_1,pos_label=1)

# PR曲线
plt.figure(figsize=(5,4), facecolor='white')
plt.plot(recall, precision, color='red')
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)
plt.title('PR曲线')


# In[ ]:


# 计算AP(直接计算线下面积)

round(np.trapz(y=precision[::-1],x=recall[::-1]),2)


# In[ ]:




