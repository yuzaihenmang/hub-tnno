# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

week2：尝试在nlpdemo中使用rnn模型训练，判断特定字符在文本中的位置。

"""



class TorchModel(nn.Module):  # 1
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=64):
        super(PositionRNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_size, 1)  # 每个位置输出一个概率值
        self.activation = torch.sigmoid
        self.loss = nn.functional.binary_cross_entropy  # 使用二元交叉熵损失
        self.sentence_length = sentence_length

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):  # 2
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        position_logits = self.classifier(rnn_out)  # (batch_size, sentence_length, 1)
        position_logits = position_logits.squeeze(-1)  # (batch_size, sentence_length)
        y_pred = self.activation(position_logits)
        # (batch_size, sen_len) -&gt; (batch_size, sen_len, vector_dim)  # 生成：文本长度乘向量
        #x = x.transpose(1, 2)  # (batch_size, sen_len, vector_dim) -&gt; (batch_size, vector_dim, sen_len)
        #x = self.pool(x)  # (batch_size, vector_dim, sen_len)-&gt;(batch_size, vector_dim, 1)
        #x = x.squeeze()  # (batch_size, vector_dim, 1) -&gt; (batch_size, vector_dim)
        #x = self.classify(x)  # (batch_size, vector_dim) -&gt; (batch_size, 1) 3*20 20*1 -&gt; 3*1
        #y_pred = self.activation(x)  # (batch_size, 1) -&gt; (batch_size, 1)

        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -&gt; [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):  # 3
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = [0] * sentence_length  # 初始化为全0
    target_chars = ["你", "我", "他"]
    for i, char in enumerate(x):
        if char in target_chars:
            y[i] = 1

# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):  # 4
    model = PositionRNNModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), 200 - sum(y)))
    correct = 0
    total = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        correct = 0
        total = 0
        for i in range(len(y_true)):
            # 对每个样本的每个位置进行预测
            for pos in range(sentence_length):
                pred = 1 if y_pred[i][pos] > 0.5 else 0
                true = y_true[i][pos].item()
                if pred == true:
                    correct += 1
                total += 1

        accuracy = correct / total
        print(f"位置预测准确率: {accuracy:.4f}")


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i]))  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ["fnvf我e", "wz你dfg", "rqwdeg", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
