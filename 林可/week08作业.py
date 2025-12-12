# -*- coding: utf-8 -*-
# 要考试了，交得迟了点
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRUCell(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size
            self.pooling_style = config["pooling_style"]
            self.triplet_margin = config.get("triplet_margin", 0.5)

    def forward(self, x):
        if self.use_bert:  # bert返回的结果是 (sequence_output, pooler_output)
            #sequence_output:batch_size, max_len, hidden_size
            #pooler_output:batch_size, hidden_size
            x = self.encoder(x)
        else:
            x = self.embedding(x)  # input shape:(batch_size, sen_len)
            x = self.encoder(x)  # input shape:(batch_size, sen_len, input_dim)
        if isinstance(x, tuple):  #RNN类的模型会同时返回隐单元向量，我们只取序列结果
                x = x[0]
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()
        x = self.feature_norm(x)
        return x

    def compute_triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        """
        计算Cosine Triplet Loss
        anchor_embeddings: 锚点样本的嵌入
        positive_embeddings: 正样本的嵌入
        negative_embeddings: 负样本的嵌入
        """
        # 计算余弦相似度
        pos_sim = F.cosine_similarity(anchor_embeddings, positive_embeddings)
        neg_sim = F.cosine_similarity(anchor_embeddings, negative_embeddings)
        losses = F.relu(-pos_sim + neg_sim + self.triplet_margin)
        return losses.mean()

    def batch_triplet_loss(self, embeddings, labels, mining_strategy="random"):
        """
        批量计算Triplet Loss
        embeddings: 所有样本的嵌入
        labels: 所有样本的标签
        mining_strategy: 三元组挖掘策略 ("random", "semihard", "hard")
        """
        batch_size = embeddings.size(0)

        # 初始化损失
        total_loss = 0
        valid_triplets = 0

        for i in range(batch_size):
            anchor = embeddings[i]
            anchor_label = labels[i]
            positive_indices = torch.where(labels == anchor_label)[0]
            positive_indices = positive_indices[positive_indices != i]
            negative_indices = torch.where(labels != anchor_label)[0]
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
            if mining_strategy == "hard":
                # 选择最难的正样本（与锚点相似度最低）
                pos_embeddings = embeddings[positive_indices]
                pos_sims = F.cosine_similarity(anchor.unsqueeze(0), pos_embeddings)
                hardest_pos_idx = torch.argmin(pos_sims)
                positive = pos_embeddings[hardest_pos_idx]
            else:
                pos_idx = positive_indices[torch.randint(0, len(positive_indices), (1,))]
                positive = embeddings[pos_idx]
            if mining_strategy == "hard":
                # 选择最难的负样本（与锚点相似度最高）
                neg_embeddings = embeddings[negative_indices]
                neg_sims = F.cosine_similarity(anchor.unsqueeze(0), neg_embeddings)
                hardest_neg_idx = torch.argmax(neg_sims)
                negative = neg_embeddings[hardest_neg_idx]
            elif mining_strategy == "semihard":
                # 选择半难的负样本（相似度介于正样本和正样本+margin之间）
                neg_embeddings = embeddings[negative_indices]
                neg_sims = F.cosine_similarity(anchor.unsqueeze(0), neg_embeddings)
                pos_sim = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))

                # 找到满足条件的负样本：pos_sim < neg_sim < pos_sim + margin
                semihard_mask = (neg_sims > pos_sim) & (neg_sims < pos_sim + self.triplet_margin)
                semihard_indices = torch.where(semihard_mask)[0]

                if len(semihard_indices) > 0:
                    semihard_idx = semihard_indices[torch.randint(0, len(semihard_indices), (1,))]
                    negative = neg_embeddings[semihard_idx]
                else:
                    # 如果没有半难负样本，使用随机负样本
                    neg_idx = negative_indices[torch.randint(0, len(negative_indices), (1,))]
                    negative = embeddings[neg_idx]
            else:  # random
                # 随机选择负样本
                neg_idx = negative_indices[torch.randint(0, len(negative_indices), (1,))]
                negative = embeddings[neg_idx]
            loss = self.compute_triplet_loss(
                anchor.unsqueeze(0),
                positive.unsqueeze(0),
                negative.unsqueeze(0)
            )

            total_loss += loss
            valid_triplets += 1
        if valid_triplets > 0:
            return total_loss / valid_triplets
        else:
            return torch.tensor(0.0, device=embeddings.device)
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1)/2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)

class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)
class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        #ModuleList类内可以放置多个模型，取用时类似于一个列表
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.num_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        #仿照bert的transformer模型结构，将self-attention替换为gcnn
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x  #通过gcnn+残差
            x = self.bn_after_gcnn[i](x)  #之后bn
            # # 仿照feed-forward层，使用两个线性层
            l1 = self.ff_liner_layers1[i](x)  #一层线性
            l1 = torch.relu(l1)  #在bert中这里是gelu
            l2 = self.ff_liner_layers2[i](l1)  #二层线性
            x = self.bn_after_ff[i](x + l2)  #残差后过bn
        return x
class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.cnn = GatedCNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x
class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x
class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        config["hidden_size"] = self.bert.config.hidden_size
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x
class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.bert.config.output_hidden_states = True

    def forward(self, x):
        layer_states = self.bert(x)[2]#(13, batch, len, hidden)
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    # 修改：添加triplet_margin参数到Config中
    Config["triplet_margin"] = 0.5
    Config["model_type"] = "bert"
    model = TorchModel(Config)

    # 测试前向传播
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    embeddings = model(x)
    print("Embeddings shape:", embeddings.shape)
    # 测试Triplet Loss计算
    labels = torch.LongTensor([0, 1])
    loss = model.batch_triplet_loss(embeddings, labels, mining_strategy="random")
    print("Triplet Loss:", loss.item())



