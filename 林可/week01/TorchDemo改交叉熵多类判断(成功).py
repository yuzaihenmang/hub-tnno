import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib  # 为解决版本不兼容而添加后端
matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'Agg' 等其他后端
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数&gt;第5个数，则为正样本，反之为负样本

week1：改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层，改为输出5维
        #self.activation = torch.sigmoid  # nn.Sigmoid() sigmoid归一化函数
        #self.loss = nn.functional.mse_loss  # loss函数采用均方差损失
        self.ce_loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -&gt; (batch_size, 1)
        #y_pred = self.ce_loss(x,y)  # (batch_size, 1) -&gt; (batch_size, 1)

        if y is not None:
            # 训练模式：计算损失
            # 确保y是正确形状
            if y.dim() > 1 and y.shape[1] == 1:
                y = y.squeeze()  # 从 (batch_size, 1) 变为 (batch_size,)
            loss = self.ce_loss(x, y)
            return loss
        else:
            # 预测模式：返回原始logits
            return x

"""
            if y is not None:
            # 训练模式：计算损失
            # 注意：y需要是形状为(batch_size,)的LongTensor，包含类别索引
            loss = self.ce_loss(x, y.squeeze())  # 使用squeeze()去掉多余的维度
            return loss
        else:
            # 预测模式：返回原始logits（或者概率）
            return 
"""

"""     if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果
"""


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():  # 一个样本
    x = np.random.random(5)
#    if x[0] > x[4]:
#        return x, 1
#    else:
#        return x, 0
    y = x.argmax()
    return x, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):  # 多个样本
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    X_array = np.array(X)  # 预先包装下x和y
    Y_array = np.array(Y)
    return torch.FloatTensor(X_array), torch.LongTensor(Y_array)  # 变为张量，float是浮点数张量，long是整数型


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    total = len(y)
    #print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        #predicted_classes = torch.max(y_pred, 1)
        _, predicted_classes = torch.max(y_pred, 1)
        # 模型预测 model.forward(x        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比


        correct = (predicted_classes == y).sum().item()
    accuracy = correct / total
    print(f"正确预测个数：{correct}, 总数：{total}, 正确率：{accuracy:.4f} ")

    return accuracy

"""            if float(y_p) < 0.5 and int(y_t) == 0:
            predicted_classes = torch.max(y_pred, 1)  # 沿着指定维度返回最大值和索引
"""


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            optim.zero_grad()  # 梯度归零  # ？？为什么要在计算损失前面
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            #optim.zero_grad()  # 梯度归零

            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        print(f"=========\n第{epoch + 1}轮平均loss:{avg_loss:.4f}")
        #print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        if (epoch+1) % 5 == 0:
            acc = evaluate(model)
            avg_loss = np.mean(watch_loss)
            log.append([acc, float(avg_loss)])


    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
"""参考代码：
    def evaluate(model, test_sample_num=100):

    评估五维向量最大值位置分类模型的准确率
    
    model.eval()
    
    # 生成测试数据
    x, y = build_dataset(test_sample_num)
    
    correct = 0
    total = len(y)
    
    with torch.no_grad():
        # 模型预测 - 输出形状: (batch_size, 5)
        y_pred = model(x)
        
        # 获取预测的类别（最大值的索引）
        _, predicted_classes = torch.max(y_pred, 1)
        
        # 与真实标签比较
        correct = (predicted_classes == y).sum().item()
    
    accuracy = correct / total
    print(f"正确预测个数：{correct}, 总数：{total}, 正确率：{accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy
"""


