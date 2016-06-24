---
title: 人工神经网络、反向传播算法和python3的简单实现
date: 2016-06-23 11:32:21
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
$$\delta\_{i}^{L}=-\frac{\partial J}{\partial z\_{i}^{L}}=-\frac{\partial J}{\partial a\_{i}^{L}}\frac{\partial a\_{i}^{L}}{\partial z\_{i}^{L}}=(y\_{i}-z\_{i}^{L}){g}'(z\_{i}^{L})=(y\_{i}-z\_{i}^{L})a^{L}\_{i}(1-a^{L}\_{i})$$
##### 向量化
$$\delta^{L}=(\mathbf{y}-\mathbf{z}^{(L)})\odot{g}'(\mathbf{z}^{(L)})=(\mathbf{y}-\mathbf{z}^{(L)})\odot\mathbf{a}^{L}\odot(1-\mathbf{a}^{L})$$
#### $L-1$层权值更新
$$ \delta^{(L-1)} = W^{(L-1)}\delta^{(L)}\odot{g}'(\mathbf{z}^{(L-1)})=W^{(L-1)}\odot(\mathbf{a}^{(L-1)}\odot(1-\mathbf{a}^{(L-1)}))$$
$$ W^{(L-1)} =W^{(L-1)}+\eta \mathbf{a}^{(L-1)}{\mathbf{\delta}^{(L)}}^{T}$$
#### $l$层权值更新
$$ \delta^{(l)} = W^{(l)}\delta^{(l+1)}\odot{g}'(\mathbf{z}^{(l)})=W^{(l)}\odot(\mathbf{a}^{(l)}\odot(1-\mathbf{a}^{(l)}))$$
$$ W^{(l)} =W^{(l)}+\eta \mathbf{a}^{(l)}{\mathbf{\delta}^{(l+1)}}^{T}$$
# python3 实现
