---
title: 人工神经网络、反向传播算法和python3的简单实现
date: 2015-09-05 11:32:21
categories:
- 机器学习
tags:
- 神经网络
- 反向传播
- python3
---
# 人工神经网络
人工神经网络（artificial neural network，缩写ANN），简称神经网络（neural network，缩写NN）或类神经网络，是一种模仿生物神经网络(动物的中枢神经系统，特别是大脑)的结构和功能的数学模型或计算模型。神经网络由大量的人工神经元联结进行计算。大多数情况下人工神经网络能在外界信息的基础上改变内部结构，是一种自适应系统。现代神经网络是一种非线性统计性数据建模工具，常用来对输入和输出间复杂的关系进行建模，或用来探索数据的模式。
## 数学模型
![人工神经网络](/upload/2/NN.png '4层人工神经网络')
### 逻辑单元
人工神经网络中，每个神经元上的逻辑单元是Sigmoid激励函数（Sigmoid activation function）或称逻辑激励函数（Logistic activation function）：
$$g(x)=\frac{1}{1+e^{-x}}$$
其导数：
$${g}'(x)=\frac{e^{-x}}{(1+e^{-x})^2}=\frac{1}{1+e^{-x}}-\frac{1}{(1+e^{-x})^2}=y(1-y)$$
### 变量定义
$a\_{i}^{(l)}$表示第$l$层第$i$个神经元（或称做激励单元（Activation Unit））的输出值。
$\mathbf{a}^{(l)}$表示$l$层输出值组成的向量。
$w^{(l)}\_{ij}$表示$l$层中的$i$神经元与$l+1$层中的$j$神经元的连接权值。
$\mathbf{w}^{(l)}\_{j}$表示$l$层中各个神经元与$l+1$层中的$j$神经元的连接权值构成的向量。
$W^{(l)}$表示第$l$层与$l+1$每条边构成的权值矩阵，$W^{(l)} \in \mathbb{R}^{\dim(\mathbf{a}^{(l)}) \times \dim(\mathbf{a}^{(l+1)})}$。
$z^{(l)}\_{i}$表示$l$层中的$i$神经元的逻辑单元输入值。$z\_{i}^{l}=\sum\_{j=1}^{\dim(\mathbf{a}^{(l-1)})}a\_{j}w\_{ji}$
$\mathbf{z}^{(l)}$表示$l$层中各个神经元的逻辑单元输入值。
## 向前传播
向前传播是指通过随机$W$来依次计算各层$\mathbf{a}$值。对于任意$l$层的$\mathbf{a}^{(l)}$可以通过$\mathbf{a}^{(l-1)}$来进行计算。
$$\because a\_{i}^{(l)} = g(z\_{i}^{(l)})$$
$$z\_{i}^{(l)}={\mathbf{a}^{(l-1)}}^{T}\mathbf{w}\_{i}^{(l-1)}$$
$$\therefore a\_{i}^{(l)} = g({\mathbf{a}^{(l-1)}}^{T}\mathbf{w}\_{i}^{(l-1)})$$
### 向量化
$$\mathbf{a}^{(l)} = g({\mathbf{a}^{(l-1)}}^{T}W^{(l-1)}) $$
## 反向传播
### 变量定义
$L$为最后一层。
$\eta$学习率。
### 误差函数
整体误差为：
$$J(W)=\frac{1}{2}\sum\_{i=1}^{\dim(\mathbf{a}^{(L)})}(a\_{i} - y\_{i})^2=\frac{1}{2} \|\| \mathbf{y} - \mathbf{a}^{(L)}  \|\|^2$$
### 梯度下降
反向传播的学习方法是基于梯度下降方法。因为权值首先被初始化为随机值，然后向误差减小的方向调整。数学表达式：
$$\Delta W= - \eta\frac{\partial J}{\partial W}$$
分量表示为：
$$\Delta w\_{ij}^{(l)} = - \eta\frac{\partial J}{\partial w\_{ij}^{(l)}}$$
权值更新为：
$$w\_{ij}^{(l)}:=w\_{ij}^{(l)} + \Delta w\_{ij}^{(l)}$$
### 变量定义
$\delta\_{i}^{(l)}=-\frac{\partial J}{\partial z\_{i}^{(l)}}$表示第$l$层第$i$个神经元敏感度（sensitive）。
$\mathbf{\delta}^{(l)}$表示第$l$层各个神经元错误构成的向量。
### 权值更新
$$w\_{ij}^{(l)}:=w\_{ij}^{(l)} + \Delta w\_{ij}^{(l)}:=w\_{ij}^{(l)} - \eta\frac{\partial J}{\partial w\_{ij}^{(l)}}:=w\_{ij}^{(l)} - \eta\frac{\partial J}{\partial z\_{j}^{(l+1)}}\frac{\partial z\_{j}^{(l+1)}}{\partial w\_{ij}^{(l)}}$$
$$\because z\_{j}^{(l+1)} = \sum\_{i=1}^{\dim(\mathbf{a}^{(l)})}a\_{j}^{(l)}w\_{ij}^{(l)}$$
$$\frac{\partial z\_{j}^{(l+1)}}{\partial w\_{ij}^{(l)}}=a\_{i}^{(l)}$$
$$\because -\frac{\partial J}{\partial z\_{j}^{(l+1)}}=\delta\_{j}^{(l+1)}$$
$$\therefore w\_{ij}^{(l)}:=w\_{ij}^{(l)} + \eta\delta\_{j}^{(l+1)} a\_{i}^{(l)}$$
#### 向量化
$$ W^{(l)} = \sum\_{i=1}^{\dim(\mathbf{a}^{(l)})}\sum\_{j=1}^{\dim(\mathbf{a}^{(l+1)})}(w\_{ij}^{(l)} + \eta\delta\_{j}^{(l+1)} a\_{i}^{(l)})=W^{(l)}+\eta \mathbf{a}^{(l)}{\mathbf{\delta}^{(l+1)}}^{T}$$
### 敏感度$\delta$
#### 一般式
$$\delta\_{i}^{(l)}=-\frac{\partial J}{\partial z\_{i}^{(l)}}=-\sum\_{j}\frac{\partial J}{\partial z\_{j}^{(l+1)}}\frac{\partial z\_{j}^{(l+1)}}{\partial a\_{i}^{(l)}}\frac{\partial a\_{i}^{(l)}}{\partial z\_{i}^{(l)}}=\sum\_{j}[\delta^{(l+1)}w\_{ij}^{(l)}]{g}'(z\_{i}^{(l)})$$
#### 向量化
$$\delta^{l}=W^{(l)}\delta^{(l+1)}\odot{g}'(\mathbf{z}^{(l)})$$
### 反向传播实现
#### $L$层的$\delta^L$
$$\delta\_{i}^{(L)}=-\frac{\partial J}{\partial z\_{i}^{(L)}}=-\frac{\partial J}{\partial a\_{i}^{(L)}}\frac{\partial a\_{i}^{(L)}}{\partial z\_{i}^{(L)}}=(y\_{i}-z\_{i}^{(L)}){g}'(z\_{i}^{(L)})=(y\_{i}-z\_{i}^{(L)})a^{(L)}\_{i}(1-a^{(L)}\_{i})$$
##### 向量化
$$\delta^{(L)}=(\mathbf{y}-\mathbf{z}^{(L)})\odot{g}'(\mathbf{z}^{(L)})=(\mathbf{y}-\mathbf{z}^{(L)})\odot\mathbf{a}^{(L)}\odot(1-\mathbf{a}^{(L)})$$
#### $L-1$层权值更新
$$ \delta^{(L-1)} = W^{(L-1)}\delta^{(L)}\odot{g}'(\mathbf{z}^{(L-1)})=W^{(L-1)}\odot(\mathbf{a}^{(L-1)}\odot(1-\mathbf{a}^{(L-1)}))$$
$$ W^{(L-1)} =W^{(L-1)}+\eta \mathbf{a}^{(L-1)}{\mathbf{\delta}^{(L)}}^{T}$$
#### $l$层权值更新
$$ \delta^{(l)} = W^{(l)}\delta^{(l+1)}\odot{g}'(\mathbf{z}^{(l)})=W^{(l)}\odot(\mathbf{a}^{(l)}\odot(1-\mathbf{a}^{(l)}))$$
$$ W^{(l)} =W^{(l)}+\eta \mathbf{a}^{(l)}{\mathbf{\delta}^{(l+1)}}^{T}$$
# python3 实现
## 超简单3层XOR网络
```python
###三层异或门python实现
import numpy as np
###训练集
X = np.array([ [0,0,1,1], [0,1,0,1] ])
y = np.array([[0,1,1,0]])
### 权值初始化
w1 = 2*np.random.random((2,3)) - 1
w2 = 2*np.random.random((3,1)) - 1
### 训练网络
for j in range(60000):
    #正向传播
    l1 = 1/(1+np.exp(-(np.dot(w1.T,X))))
    l2 = 1/(1+np.exp(-(np.dot(w2.T,l1))))
    #反向传播
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = w2.dot(l2_delta) * (l1 * (1-l1))
    #权值更新
    w2 += l1.dot(l2_delta.T)
    w1 += X.dot(l1_delta.T)
```
## 通用人工神经网络实现
```python
import numpy as np
#Sigmoid函数
def Sigmoid(x):
    y = 1/(1+np.exp(-x))
    return(y)
class NN:
    def __init__(self,X,Y,layer,eta):
    '''NN(输入，y，网络结构列表，学习率)'''
        self.X,self.Y,self.layer,self.eta=(X,Y,layer,eta)
    def train(self,times):
        layerList = self.layer ##NN结构
        self.ErrHis = []  #误差历史
        weight = [] #权值
        delta=[]  #敏感度
        #权值初始化
        for l in range(1,len(layerList)):
            weight.append(2*np.random.random((layerList[l-1],layerList[l])) - 1)
        #学习
        for i in range(times):
            #初始化
            activation = []
            activation.append(self.X)
            #向前传播，获取激励
            for l in range(len(weight)):
                activation.append(Sigmoid(np.dot(weight[l].T,activation[l])))
            #误差
            err = 1/2 * sum((self.Y.T - activation[-1].T)**2)
            self.ErrHis.append(err[0])
            #敏感度反向传播
            if delta: PreDelta = delta #上次敏感度保留
            delta=[]  #初始化
            A = [i for i in range(len(activation))] #反向列队
            A.remove(0)
            A.reverse()
            for a in A:
                if not delta: #最后一层
                    delta.append((self.Y-activation[a])*(activation[a]*(1-activation[a])))
                else: #其他层
                    delta.append(weight[a].dot(delta[-1])*(activation[a]*(1-activation[a])))
            delta.reverse() #正向化
            PreWeight =weight #上次权值保留
            #权值更新
            for l in range(len(weight)):
                weight[l] += self.eta*activation[l].dot(delta[l].T)
        #结束
        self.activation = activation
        self.delta = PreDelta
        self.weight = PreWeight
    def predict(self,TEST): #预测
        if not self.weight:
            print('untrained NN')
            return False
        else:
            PredictA=[]
            PredictA.append(TEST)
            for l in range(len(self.weight)):
                PredictA.append(Sigmoid(np.dot(self.weight[l].T,PredictA[l])))
            return(PredictA[-1])
    def weight(self): #返回权值
        return(self.weight)      
    def errHis(self): #返回误差历史
        return(self.ErrHis)
```
## 带偏置（Bias）节点的神经网络
带偏置的神经网络，在第1层到$L-1$层都有着一个偏置节点；偏移节点有个特定，其只与下一层的非偏移节点相连。这样，$$\mathbf{w}^{(l)}的维度=l层所有节点数\times(l+1)层非偏置节点数$$
![带偏置人工神经网络](/upload/2/NN_with_bias.png '4层带偏置人工神经网络')
正向传播时:$$\mathbf{a}^{(l+1)}\_{no bias} = g(\mathbf{w}^{(l)}\mathbf{a}^{(l)}\_{bias})$$
反向传播时，第$L$层敏感度不变：$$\delta^{(L)}=(\mathbf{y}-\mathbf{z}^{(L)})\odot{g}'(\mathbf{z}^{(L)})=(\mathbf{y}-\mathbf{z}^{(L)})\odot\mathbf{a}^{L}\odot(1-\mathbf{a}^{(L)})$$
第$L-1$层敏感度：$$\delta^{(L-1)}=W^{(L-1)}\delta^{(L)}\odot{g}'(\mathbf{z}^{(L-1)})$$
第$l$层的敏感度：$$\delta^{(l)}=W^{(l)}\delta^{(l)}\_{no bias}\odot{g}'(\mathbf{z}^{(l)})$$
权值更新：$$W^{(l)} =W^{(l)}+\eta \mathbf{a}^{(l)}{\mathbf{\delta}^{(l+1)}}\_{nobias}^{T}$$
## python3实现
```python
import numpy as np
def Sigmoid(x):
    y = 1/(1+np.exp(-x))
    return(y)
class NN:
    def __init__(self,X,Y,layer,eta):
        self.X,self.Y,self.layer,self.eta=(X,Y,layer,eta)
    def train(self,times):
        layerList = self.layer
        self.ErrHis = []
        weight = []
        delta=[]
        #权值初始化
        for l in range(1,len(layerList)):
            weight.append(2*np.random.random((layerList[l-1]+1,layerList[l])) - 1)
        #学习
        for i in range(times):
            #初始化
            activation = []
            activation.append(self.X)
            biasA = np.array([np.repeat(1,np.shape(X)[1])]) #bias
            #向前传播，获取激励
            for l in range(len(weight)):
                activation.append(Sigmoid(np.dot(weight[l].T,np.concatenate((biasA,activation[l])))))
            #误差
            err = 1/2 * sum((self.Y.T - activation[-1].T)**2)
            self.ErrHis.append(err[0])
            #敏感度反向传播
            if delta: PreDelta = delta
            delta=[]
            A = [i for i in range(len(activation))]
            A.remove(0)
            A.reverse()
            for a in A:
                if not delta: #最后一层
                    delta.append((self.Y-activation[a])*(activation[a]*(1-activation[a])))
                elif len(delta) == 1: #倒数第二层
                        delta.append(weight[a].dot(delta[-1])*(np.concatenate((biasA,activation[a]))*(1-np.concatenate((biasA,activation[a])))))
                else: #其他层
                    delta.append(weight[a].dot(delta[-1][1:,:])*(np.concatenate((biasA,activation[a]))*(1-np.concatenate((biasA,activation[a])))))
            #上一次权值保存，及权值更新
            delta.reverse()
            PreWeight =weight
            for l in range(len(weight)):
                if l == len(weight)-1:
                    weight[l] += self.eta*np.concatenate((biasA,activation[l])).dot(delta[l].T)
                else:
                    weight[l] += self.eta*np.concatenate((biasA,activation[l])).dot(delta[l][1:,:].T)
        #结束
        self.activation = activation
        self.delta = PreDelta
        self.weight = PreWeight
    def predict(self,TEST):
        if not self.weight:
            print('untrained NN')
            return False
        else:
            PredictA=[]
            biasA = np.array([np.repeat(1,np.shape(TEST)[1])])
            PredictA.append(TEST)
            for l in range(len(self.weight)):
                PredictA.append(Sigmoid(np.dot(self.weight[l].T,np.concatenate((PredictA[l],biasA)))))
            print(PredictA)
            return(PredictA[-1])
    def weight(self):
        return(self.weight)
    def errHis(self):
        return(self.ErrHis)
```
