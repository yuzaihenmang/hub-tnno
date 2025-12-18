import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)多分类任务
规律：正确类别为最大的那个类

"""



class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,5)
        # self.activation = nn.Softmax(dim=5)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = nn.Softmax(dim=1)(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return y_pred



# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(5)                             # x = torch.rand(5) 可以
    return x, x.argmax()


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # torch.tensor(Y) 没用






# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    correct, wrong = 0, 0
    x, y = build_dataset(test_sample_num)
    with torch.no_grad():
        y_pred = model(x)
        # y_pred = TorchModel(5).forward(x) 没用
        for y_p, y_t in zip(y_pred, y):
            if y_p.argmax() == y_t:
                correct += 1
            else:
                wrong += 1
        print("正确个数为：%d，正确率为：%f"%(correct, correct/(correct + wrong)))
        return correct / (correct + wrong)


def main():

    epoch_num = 100
    batch_size = 50
    train_sample_num = 5000
    input_size = 5
    learning_rate = 0.05
    model = TorchModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample_num)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(train_sample_num//batch_size):
            x = train_x[batch_size*i:batch_size*(i+1)]
            y = train_y[batch_size*i:batch_size*(i+1)]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print('——————————————\n第%d轮，loss是%f'%(epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), 'model.pt')
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label = 'acc')
    plt.plot(range(len(log)), [l[1] for l in log], label = 'loss')
    plt.legend()
    plt.show()
    return



# 使用训练好的模型做预测
def predict(model_path, input_vec):
    pass

if __name__ == "__main__":
    main()