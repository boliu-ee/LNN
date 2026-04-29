import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from ncps.torch import LTC

# ----------------------------- 0. 全局设置 -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ----------------------------- 1. 数据生成 -----------------------------
def gen_sine(freq, duration, dt=0.05, noise=0.01):
    t = np.arange(0, duration, dt)
    x = np.sin(2 * np.pi * freq * t) + noise * np.random.randn(len(t))
    return t, x.astype(np.float32)

t_train, x_train = gen_sine(0.5, duration=100)
_, x1 = gen_sine(0.5, 20)
_, x2 = gen_sine(2.0, 20)
_, x3 = gen_sine(1.0, 20)
_, x4 = gen_sine(0.2, 20)
x_test = np.concatenate([x1, x2, x3, x4])
t_test = np.arange(len(x_test)) * 0.05

mean, std = x_train.mean(), x_train.std()
x_train_norm = (x_train - mean) / std
x_test_norm = (x_test - mean) / std

window_size = 30
def create_dataset(seq, window_size):
    X, y = [], []
    for i in range(len(seq) - window_size):
        X.append(seq[i:i+window_size])
        y.append(seq[i+window_size])
    return np.array(X), np.array(y).reshape(-1, 1)

X_train_norm, y_train_norm = create_dataset(x_train_norm, window_size)
X_train_t = torch.tensor(X_train_norm[..., np.newaxis], dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_norm, dtype=torch.float32).to(device)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# ----------------------------- 2. 工具：参数量计算 & 自动对齐 -----------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

TARGET_PARAMS = 1233  # 以16单元LSTM为基准

# ---------- FNN ----------
class FNN(nn.Module):
    def __init__(self, window_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

# ---------- RNN ----------
class SimpleRNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x, h0=None):
        out, hn = self.rnn(x, h0)
        return self.fc(out[:, -1, :])

# ---------- LSTM ----------
class LSTMModel(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x, states=None):
        out, (hn, cn) = self.lstm(x, states)
        return self.fc(out[:, -1, :])

# ---------- LTC ----------
class LTCModel(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.ltc = LTC(input_size=1, units=units, return_sequences=False, batch_first=True)
        self.fc = nn.Linear(units, 1)
    def forward(self, x, hx=None):
        out, hx = self.ltc(x, hx)
        return self.fc(out)

# 自动寻找LTC最优units
best_ltc_units = 16
best_diff = float('inf')
for u in range(10, 51):
    test_model = LTCModel(u).to(device)
    params = count_params(test_model)
    diff = abs(params - TARGET_PARAMS)
    if diff < best_diff:
        best_diff = diff
        best_ltc_units = u
print(f"LTC 自动选择的单元数: {best_ltc_units}，参数量差: {best_diff}")

# FNN 和 RNN 的最优 hidden_size
fnn_hidden = round((TARGET_PARAMS - 1) / 32)       # ≈39
a, b, c = 1, 4, 1 - TARGET_PARAMS
discriminant = b**2 - 4*a*c
rnn_hidden = int((-b + discriminant**0.5) / (2*a))  # ≈34
print(f"FNN hidden: {fnn_hidden}, RNN hidden: {rnn_hidden}")

# ----------------------------- 3. 初始化模型 -----------------------------
models = {
    'FNN': FNN(window_size, fnn_hidden).to(device),
    'RNN': SimpleRNN(rnn_hidden).to(device),
    'LSTM': LSTMModel().to(device),          # 16 单元
    'LTC': LTCModel(best_ltc_units).to(device)
}

# 打印参数量
print("\n====== 模型参数量对比 ======")
for name, model in models.items():
    p = count_params(model)
    print(f"{name:>6}: {p:>6} 参数")
print("==========================\n")

# ----------------------------- 4. 训练 -----------------------------
history = {}
for name, model in models.items():
    print(f'训练 {name} ...')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(20):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if epoch % 5 == 0:
            print(f'  Epoch {epoch:2d} | Loss: {avg_loss:.6f}')
    history[name] = losses
    print(f'  最终 Loss: {losses[-1]:.6f}\n')

# ----------------------------- 5. 在线预测测试 (修正隐藏状态形状) -----------------------------
def online_predict_recurrent(model, seq_norm, window_size, cell_type):
    n = len(seq_norm)
    preds = np.full(n, np.nan, dtype=np.float32)          # 长度 = 真实序列，缺失值用 NaN

    # 初始化隐藏状态
    if cell_type == 'rnn':
        h = torch.zeros(1, 1, model.rnn.hidden_size).to(device)
    elif cell_type == 'lstm':
        h = (torch.zeros(1, 1, model.lstm.hidden_size).to(device),
             torch.zeros(1, 1, model.lstm.hidden_size).to(device))
    elif cell_type == 'ltc':
        h = torch.zeros(1, model.ltc.state_size).to(device)

    # 预热：用前 window_size 个点更新隐藏状态（不产生预测）
    x_pre = torch.tensor(seq_norm[:window_size], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        if cell_type == 'rnn':
            _, h = model.rnn(x_pre, h)
        elif cell_type == 'lstm':
            _, h = model.lstm(x_pre, h)
        elif cell_type == 'ltc':
            _, h = model.ltc(x_pre, h)

    # 逐点预测：用 x_t 预测 x_{t+1}
    with torch.no_grad():
        for t in range(window_size, n-1):
            x_t = torch.tensor(seq_norm[t], dtype=torch.float32).reshape(1, 1, 1).to(device)
            if cell_type == 'rnn':
                out, h = model.rnn(x_t, h)
            elif cell_type == 'lstm':
                out, h = model.lstm(x_t, h)
            elif cell_type == 'ltc':
                out, h = model.ltc(x_t, h)
            preds[t+1] = model.fc(out).cpu().numpy().item()   # 预测值存入 t+1
    return preds


def online_predict_fnn(model, seq_norm, window_size):
    n = len(seq_norm)
    preds = np.full(n, np.nan, dtype=np.float32)
    with torch.no_grad():
        for t in range(window_size, n-1):
            # 窗口取 [t-window_size, t-1] 来预测 x_t
            win = torch.tensor(seq_norm[t-window_size:t], dtype=torch.float32).reshape(1, window_size).to(device)
            preds[t] = model(win).cpu().numpy().item()
    return preds

# 计算各模型预测结果
preds_fnn = online_predict_fnn(models['FNN'], x_test_norm, window_size)
preds_rnn = online_predict_recurrent(models['RNN'], x_test_norm, window_size, 'rnn')
preds_lstm = online_predict_recurrent(models['LSTM'], x_test_norm, window_size, 'lstm')
preds_ltc = online_predict_recurrent(models['LTC'], x_test_norm, window_size, 'ltc')

def denorm(y):
    return y * std + mean

preds_fnn_orig = denorm(preds_fnn)
preds_rnn_orig = denorm(preds_rnn)
preds_lstm_orig = denorm(preds_lstm)
preds_ltc_orig = denorm(preds_ltc)
x_test_orig = x_test

# ----------------------------- 6. 可视化 -----------------------------
def plot_predictions(ax, t, y_true, pred, name):
    ax.plot(t, y_true, 'gray', linewidth=1, label='True')
    ax.plot(t, pred, linewidth=1, label=name)
    ax.set_title(name)
    ax.legend()
    ax.set_ylim(-2, 2)

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
plot_predictions(axes[0], t_test, x_test_orig, preds_fnn_orig, 'FNN')
plot_predictions(axes[1], t_test, x_test_orig, preds_rnn_orig, 'RNN')
plot_predictions(axes[2], t_test, x_test_orig, preds_lstm_orig, 'LSTM')
plot_predictions(axes[3], t_test, x_test_orig, preds_ltc_orig, 'LTC')
plt.tight_layout()

def sliding_mse(y_true, y_pred, window=50):
    mse = []
    for i in range(window, len(y_true)-1):
        mse.append(np.mean((y_true[i-window:i]-y_pred[i-window:i])**2))
    return np.array(mse)

fig2, ax2 = plt.subplots(figsize=(12, 3))
for name, pred in [('FNN', preds_fnn_orig), ('RNN', preds_rnn_orig),
                   ('LSTM', preds_lstm_orig), ('LTC', preds_ltc_orig)]:
    mse = sliding_mse(x_test_orig, pred)
    ax2.plot(t_test[51:], mse, label=name)
ax2.axvline(20, color='gray', linestyle='--', label='freq: 0.5→2Hz')
ax2.axvline(40, color='gray', linestyle=':', label='freq: 2→1Hz')
ax2.axvline(60, color='gray', linestyle='-.', label='freq: 1→0.2Hz')
ax2.set_xlabel('Time')
ax2.set_ylabel('Sliding MSE')
ax2.legend()
ax2.set_title('Online Prediction Error (equal-parameter setting)')
plt.tight_layout()
plt.show()
