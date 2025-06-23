import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(LSTMAutoencoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # 编码器和解码器
        self.encoder = CustomLSTMLayer(input_size, hidden_size)
        self.decoder = CustomLSTMLayer(hidden_size, hidden_size)
        self.reconstruct = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 初始化隐藏状态
        h_enc, c_enc = self.init_hidden(batch_size)

        # 编码器前向传播
        # encoder_outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]
            h_enc, c_enc = self.encoder(input_t, h_enc, c_enc)
            # encoder_outputs.append(h_enc)

        # 解码器前向传播
        decoder_outputs = []
        h_dec = h_enc
        c_dec = c_enc
        decoder_input = h_dec

        for t in range(seq_len):
            h_dec, c_dec = self.decoder(decoder_input, h_dec, c_dec)
            output_t = self.reconstruct(h_dec)
            decoder_outputs.append(output_t)
            decoder_input = h_dec

        # 返回完整重构序列
        reconstructed = torch.stack(decoder_outputs, dim=1)
        return reconstructed

    def init_hidden(self, batch_size):
        # 初始化隐藏状态和细胞状态
        h = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(batch_size, self.hidden_size, device=self.device)
        return h, c


class CustomLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 遗忘门参数
        self.W_f = nn.Linear(input_size, hidden_size)
        self.V_f = nn.Linear(hidden_size, hidden_size)
        self.b_f = nn.Parameter(torch.ones(hidden_size))  # 初始化为1，而不是0

        # 输入门参数
        self.W_i = nn.Linear(input_size, hidden_size)
        self.V_i = nn.Linear(hidden_size, hidden_size)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        # 输出门参数
        self.W_o = nn.Linear(input_size, hidden_size)
        self.V_o = nn.Linear(hidden_size, hidden_size)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        # 细胞状态参数
        self.W_c = nn.Linear(input_size, hidden_size)
        self.V_c = nn.Linear(hidden_size, hidden_size)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        # 初始化权重
        self._initialize_weights()

    def forward(self, x, h_prev, c_prev):
        # 计算遗忘门
        f_t = torch.sigmoid(self.W_f(x) + self.V_f(h_prev) + self.b_f)
        # 计算输入门
        i_t = torch.sigmoid(self.W_i(x) + self.V_i(h_prev) + self.b_i)
        # 计算输出门
        o_t = torch.sigmoid(self.W_o(x) + self.V_o(h_prev) + self.b_o)
        # 计算细胞状态候选值
        c_tilde = torch.tanh(self.W_c(x) + self.V_c(h_prev) + self.b_c)
        # 更新细胞状态
        c_next = (f_t * c_prev) + (i_t * c_tilde)
        # 计算隐藏状态
        h_next = o_t * torch.tanh(c_next)
        return h_next, c_next

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                # 使用xavier初始化权重
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name:
                if 'b_f' in name:
                    # 遗忘门偏置初始化为1，有助于保持长期记忆
                    nn.init.constant_(param, 1.0)
                elif 'b_i' in name:
                    # 输入门偏置初始化为0
                    nn.init.constant_(param, 0.0)
                elif 'b_o' in name:
                    # 输出门偏置初始化为0
                    nn.init.constant_(param, 0.0)
                elif 'b_c' in name:
                    # 细胞状态偏置初始化为0
                    nn.init.constant_(param, 0.0)


# 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    model.to(device)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        batch_count = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.8f}")
    return model
