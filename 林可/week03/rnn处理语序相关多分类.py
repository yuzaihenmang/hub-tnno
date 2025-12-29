"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp六分类任务：与语序相关
规律：‘我’出现在哪为哪类，‘我’没出现为第6类

"""
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()    ############
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, sentence_length+1, bias=True, batch_first=True)  # 六分类就填7，总是比文本数多一，多了个“‘没’我的情况”
        # self.classify = nn.Linear(vector_dim, 7)
        self.activation = nn.Softmax(dim=-1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        output, x = self.rnn(x)
        x = np.squeeze(x)
        # x = self.classify(x)
        y_pred = self.activation(x)
        if y is not None:
            loss = self.loss(x, y)
            return loss
        else:
            return y_pred

def build_vocab():
    strs = '你我他defghijklmnopqrstuvwxyz'
    vocab = {'[pad]':0}
    for i, char in enumerate(strs):
        vocab[char] = i + 1
    vocab['[unk]'] = len(vocab)
    return vocab

def date_simple(vocab, sentence_lenght):
    x = []
    strs = list(vocab.keys())
    chars = [np.random.choice(strs) for _ in range(sentence_lenght)]
    for char in chars:
        encode = vocab.get(char, vocab['[unk]'])
        x.append(encode)
    if '我' in chars:
        for i, str in enumerate(chars):
            if str == '我':
                return x, i
    else:
        return x, 6
def build_dateset(nums, vocab, sentence_lenght):
    X = []
    Y = []
    for num in range(nums):
        x, y = date_simple(vocab, sentence_lenght)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X), np.array(Y)
    # print(X, Y)
    return torch.LongTensor(X), torch.LongTensor(Y)

def build_model(vocab, sentence_length, vector_dim):
    model = TorchModel(vector_dim, sentence_length, vocab)
    return model

def evaluate(model, vocab, sentence_lenght):
    model.eval()
    test_nums = 50
    x, y = build_dateset(test_nums, vocab, sentence_lenght)
    correct, wrong = 0, 0
    with torch.no_grad():       #################
        y_pred = model(x)   #先不计算梯度，再计算
        for y_p, y_t in zip(y_pred, y):
            # print(torch.round(y_p), y_t)
            if np.argmax(np.array(y_p)) == y_t:
                correct += 1
            else:
                wrong += 1
    return correct / (correct + wrong)


def main():
    epoch_count = 20
    train_nums = 1000
    batch_size = 20
    learning_rate = 0.01
    sentence_lenght = 6
    vector_dim = 20
    log = []
    vocab = build_vocab()
    model = build_model(vocab, sentence_lenght, vector_dim)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epoch_count):
        model.train()
        watch_loss = []
        for i in range(train_nums//batch_size):
            X, Y = build_dateset(batch_size, vocab, sentence_lenght)
            model.zero_grad()
            loss = model(X, Y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print('第%d轮的loss为：%f'%(epoch+1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_lenght)
        log.append([acc, np.mean(watch_loss)])
    torch.save(model.state_dict(), 'model.bin')
    writer = open('vocab.json', 'w', encoding='utf-8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    plt.plot(range(len(log)), [l[0] for l in log], label='acc')
    plt.plot(range(len(log)),[l[1] for l in log], label='loss')
    plt.legend()
    plt.show()

def predict(model_path, vocab_path, test_strings):
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    model = TorchModel(20, 6, vocab)
    model.load_state_dict(torch.load(model_path))
    X = []
    for strs in test_strings:
        encode = []
        for str in strs:
            x = vocab.get(str, vocab['[unk]'])
            encode.append(x)
        X.append(encode)
    X = torch.LongTensor(X)
    # print('X:', X)
    # print('model.state_dict:', model.state_dict())
    model.eval()
    with torch.no_grad():
        result = model.forward(X)
        for vec, res in zip(test_strings, result):
            print(f"输入\'{vec}\'，预测为{np.argmax(np.array(res))}，概率为{res}")  # %(vec, np.argmax(np.array(res)), res)失败
            # print(f"输入\'%s\'，预测为%d，概率为%f"%(vec, np.array(np.argmax(np.array(res))), res))

if __name__ == '__main__':
    main()
    test_strings = ["fnvf我a", "wz你dfg", "rqwdeg", "n我kwww"]
    predict('model.bin', 'vocab.json', test_strings)
