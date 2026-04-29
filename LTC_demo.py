import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json
import os
from ncps.torch import LTC

# ================== 0. 全局设置 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = "LTC_demo_outputs"  # 输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================== 1. 数据生成 ==================
def gen_sine(freq, duration, dt=0.05, noise=0.01):
    t = np.arange(0, duration, dt)
    x = np.sin(2 * np.pi * freq * t) + noise * np.random.randn(len(t))
    return t, x.astype(np.float32)


# 训练集
t_train, x_train = gen_sine(0.5, duration=100)

# 测试集（分段频率）
_, x1 = gen_sine(0.5, 20)
_, x2 = gen_sine(2.0, 20)
_, x3 = gen_sine(1.0, 20)
_, x4 = gen_sine(0.2, 20)
x_test = np.concatenate([x1, x2, x3, x4])
t_test = np.arange(len(x_test)) * 0.05

# 归一化
mean, std = x_train.mean(), x_train.std()
x_train_norm = (x_train - mean) / std
x_test_norm = (x_test - mean) / std

window_size = 30


def create_dataset(seq, window_size):
    X, y = [], []
    for i in range(len(seq) - window_size):
        X.append(seq[i:i + window_size])
        y.append(seq[i + window_size])
    return np.array(X), np.array(y).reshape(-1, 1)


X_train_norm, y_train_norm = create_dataset(x_train_norm, window_size)
X_train_t = torch.tensor(X_train_norm[..., np.newaxis], dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train_norm, dtype=torch.float32).to(device)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)


# ================== 2. 工具函数 ==================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


TARGET_PARAMS = 1233  # 以16单元LSTM为基准


# ---------- 模型定义 ----------
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


class SimpleRNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None):
        out, hn = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, states=None):
        out, (hn, cn) = self.lstm(x, states)
        return self.fc(out[:, -1, :])


class LTCModel(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.ltc = LTC(input_size=1, units=units, return_sequences=False, batch_first=True)
        self.fc = nn.Linear(units, 1)

    def forward(self, x, hx=None):
        out, hx = self.ltc(x, hx)
        return self.fc(out)


# 自动选择LTC单元数
best_ltc_units = 16
best_diff = float('inf')
for u in range(10, 51):
    test_model = LTCModel(u).to(device)
    diff = abs(count_params(test_model) - TARGET_PARAMS)
    if diff < best_diff:
        best_diff = diff
        best_ltc_units = u
print(f"LTC units = {best_ltc_units}, param diff = {best_diff}")

# FNN & RNN 最优尺寸
fnn_hidden = round((TARGET_PARAMS - 1) / 32)
a, b, c = 1, 4, 1 - TARGET_PARAMS
rnn_hidden = int((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))
print(f"FNN hidden = {fnn_hidden}, RNN hidden = {rnn_hidden}")

# ================== 3. 初始化模型 ==================
models = {
    'FNN': FNN(window_size, fnn_hidden).to(device),
    'RNN': SimpleRNN(rnn_hidden).to(device),
    'LSTM': LSTMModel().to(device),
    'LTC': LTCModel(best_ltc_units).to(device)
}

param_counts = {name: count_params(model) for name, model in models.items()}
print("\n====== 参数量 ======")
for k, v in param_counts.items():
    print(f"{k:>6}: {v}")
print("====================\n")

# ================== 4. 训练 ==================
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


# ================== 5. 在线预测函数（带tau收集） ==================
def online_predict_recurrent(model, seq_norm, window_size, cell_type, ltc_tau_out=None):
    def _sigmoid(v, mu, sigma):
        """LTC 内部使用的电导激活函数（对数域安全）"""
        return torch.sigmoid((v - mu) / (sigma + 1e-8))

    n = len(seq_norm)
    preds = np.full(n, np.nan, dtype=np.float32)

    # 初始化状态
    if cell_type == 'rnn':
        h = torch.zeros(1, 1, model.rnn.hidden_size).to(device)
    elif cell_type == 'lstm':
        h = (torch.zeros(1, 1, model.lstm.hidden_size).to(device),
             torch.zeros(1, 1, model.lstm.hidden_size).to(device))
    elif cell_type == 'ltc':
        h = torch.zeros(1, model.ltc.state_size).to(device)

    # 预热
    x_pre = torch.tensor(seq_norm[:window_size], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        if cell_type == 'rnn':
            _, h = model.rnn(x_pre, h)
        elif cell_type == 'lstm':
            _, h = model.lstm(x_pre, h)
        elif cell_type == 'ltc':
            _, h = model.ltc(x_pre, h)

    with torch.no_grad():
        for t in range(window_size, n - 1):
            x_t = torch.tensor(seq_norm[t], dtype=torch.float32).reshape(1, 1, 1).to(device)
            if cell_type == 'rnn':
                out, h = model.rnn(x_t, h)
            elif cell_type == 'lstm':
                out, h = model.lstm(x_t, h)
            elif cell_type == 'ltc':
                out, h = model.ltc(x_t, h)
                # 安全获取 tau
                if ltc_tau_out is not None:
                    # 取自 model.ltc.rnn_cell 的参数（与 forward 所用一致）
                    cell = model.ltc.rnn_cell
                    cm = cell.cm.abs()  # [H]
                    gleak = cell.gleak.abs()  # [H]
                    eps = cell._epsilon

                    # 感觉电导 (C=1, 探测显示参数形状为 [1,16] )
                    sensory_w = cell.sensory_w.view(1, -1)  # [1, H] -> 调整为 [H] 或 [H,1]
                    sensory_mu = cell.sensory_mu.view(1, -1)
                    sensory_sigma = cell.sensory_sigma.view(1, -1)
                    # x_t 形状 [1,1,1] 取 [0,0,:] -> [1]
                    x_val = x_t[0, 0, :]  # [1]
                    s_act = sensory_w * _sigmoid(x_val, sensory_mu, sensory_sigma)  # [1,H]
                    sensory_g = s_act.sum(dim=0)  # [H]

                    # 递归电导
                    rec_w = cell.w  # [H,H]
                    rec_mu = cell.mu
                    rec_sigma = cell.sigma
                    # h 形状 [1, H] (batch_first 模式下的隐藏状态)
                    h_current = h if isinstance(h, torch.Tensor) else h[0]  # LTC 返回 h 为张量
                    h_expand = h_current.unsqueeze(0)  # [1,1,H]
                    mu_expand = rec_mu.unsqueeze(0)  # [1,H,H]
                    sigma_expand = rec_sigma.unsqueeze(0)
                    w_expand = rec_w.unsqueeze(0)
                    rec_act = w_expand * _sigmoid(h_expand, mu_expand, sigma_expand)  # [1,H,H]
                    rec_g = rec_act.sum(dim=-1).squeeze(0)  # [H]

                    g_total = gleak + sensory_g + rec_g + eps
                    tau_now = (cm / g_total).detach().cpu().numpy()  # [H]
                    ltc_tau_out.append(tau_now)
            preds[t + 1] = model.fc(out).cpu().numpy().item()
    return preds


def online_predict_fnn(model, seq_norm, window_size):
    n = len(seq_norm)
    preds = np.full(n, np.nan, dtype=np.float32)
    with torch.no_grad():
        for t in range(window_size, n - 1):
            win = torch.tensor(seq_norm[t - window_size:t], dtype=torch.float32).reshape(1, window_size).to(device)
            preds[t] = model(win).cpu().numpy().item()
    return preds


# ================== 6. 计算预测 ==================
ltc_tau_list = []  # 收集LTC的 tau 向量（每一步的形状 (units,)）

preds_fnn = online_predict_fnn(models['FNN'], x_test_norm, window_size)
preds_rnn = online_predict_recurrent(models['RNN'], x_test_norm, window_size, 'rnn')
preds_lstm = online_predict_recurrent(models['LSTM'], x_test_norm, window_size, 'lstm')
preds_ltc = online_predict_recurrent(models['LTC'], x_test_norm, window_size, 'ltc',
                                     ltc_tau_out=ltc_tau_list)


# 反归一化
def denorm(y):
    return y * std + mean


preds_fnn_orig = denorm(preds_fnn)
preds_rnn_orig = denorm(preds_rnn)
preds_lstm_orig = denorm(preds_lstm)
preds_ltc_orig = denorm(preds_ltc)
x_test_orig = x_test


# ================== 7. 计算测试集MSE（忽略NaN） ==================
def compute_mse(y_true, y_pred):
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if np.sum(mask) == 0:
        return float('inf')
    return np.mean((y_true[mask] - y_pred[mask]) ** 2)


test_mse = {
    'FNN': compute_mse(x_test_orig, preds_fnn_orig),
    'RNN': compute_mse(x_test_orig, preds_rnn_orig),
    'LSTM': compute_mse(x_test_orig, preds_lstm_orig),
    'LTC': compute_mse(x_test_orig, preds_ltc_orig)
}

# ================== 8. 保存JSON结果 ==================
results = {
    'window_size': window_size,
    'training_frequency': 0.5,
    'test_frequency_segments': [0.5, 2.0, 1.0, 0.2],
    'target_params': TARGET_PARAMS,
    'model_params': param_counts,
    'final_train_loss': {name: history[name][-1] for name in models},
    'test_mse': test_mse
}

json_path = os.path.join(OUTPUT_DIR, "results.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"结果已保存至 {json_path}")

# ================== 9. 绘图并保存 ==================

# ---- 图1：训练波形 ----
fig1, ax1 = plt.subplots(figsize=(10, 3))
ax1.plot(t_train, x_train, linewidth=0.6)
ax1.set_title('Training signal (0.5 Hz sine + noise)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
fig1.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "training_waveform.png"), dpi=150)
plt.close(fig1)

# ---- 图2：测试波形（分段频率） ----
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(t_test, x_test_orig, linewidth=0.6)
ax2.set_title('Test signal (frequency jumps)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude')
for t_change in [20, 40, 60]:
    ax2.axvline(t_change, color='red', linestyle='--', alpha=0.7)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "test_waveform.png"), dpi=150)
plt.close(fig2)

# ---- 图3：对比各模型预测 ----
fig3, axes3 = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
plot_items = [('FNN', preds_fnn_orig), ('RNN', preds_rnn_orig),
              ('LSTM', preds_lstm_orig), ('LTC', preds_ltc_orig)]
for ax, (name, pred) in zip(axes3, plot_items):
    ax.plot(t_test, x_test_orig, 'gray', linewidth=1, label='True')
    ax.plot(t_test, pred, linewidth=1, label=name)
    ax.set_title(name)
    ax.legend()
    ax.set_ylim(-2, 2)
fig3.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "prediction_comparison.png"), dpi=150)
plt.close(fig3)


# ---- 图4：滑动MSE ----
def sliding_mse(y_true, y_pred, window=50):
    mse = []
    for i in range(window, len(y_true) - 1):
        mse.append(np.mean((y_true[i - window:i] - y_pred[i - window:i]) ** 2))
    return np.array(mse)


fig4, ax4 = plt.subplots(figsize=(12, 3))
for name, pred in plot_items:
    mse = sliding_mse(x_test_orig, pred)
    ax4.plot(t_test[51:], mse, label=name)
ax4.axvline(20, color='gray', linestyle='--', label='0.5→2Hz')
ax4.axvline(40, color='gray', linestyle=':', label='2→1Hz')
ax4.axvline(60, color='gray', linestyle='-.', label='1→0.2Hz')
ax4.set_xlabel('Time')
ax4.set_ylabel('Sliding MSE')
ax4.legend()
ax4.set_title('Online Prediction Error (equal-param)')
fig4.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, "sliding_mse.png"), dpi=150)
plt.close(fig4)

# ---- 图5：LTC有效时间常数 tau ----
if ltc_tau_list:
    tau_array = np.array(ltc_tau_list)  # [steps, H]
    num_steps = tau_array.shape[0]
    tau_times = t_test[window_size: window_size + num_steps]

    fig5, ax5 = plt.subplots(figsize=(12, 4))
    for neu in range(tau_array.shape[1]):
        ax5.plot(tau_times, tau_array[:, neu], alpha=0.3, color='blue', linewidth=0.5)
    mean_tau = tau_array.mean(axis=1)
    ax5.plot(tau_times, mean_tau, 'red', linewidth=2, label='Mean τ')
    ax5.set_title('LTC Effective Time Constant τ (computed from ODE params)')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('τ')
    ax5.legend()
    for t_change in [20, 40, 60]:
        ax5.axvline(t_change, color='gray', linestyle='--', alpha=0.5)
    fig5.tight_layout()
    fig5.savefig(os.path.join(OUTPUT_DIR, "ltc_tau_dynamics.png"), dpi=150)
    plt.close(fig5)
    print(f"τ 图已保存，共 {num_steps} 步")
else:
    print("未收集到 tau 数据")

print(f"\n所有输出已保存至文件夹：{OUTPUT_DIR}/")
