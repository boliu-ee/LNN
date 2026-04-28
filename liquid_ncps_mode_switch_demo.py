import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ncps.torch import LTC


# ============================================================
# 1. Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# ============================================================
# 2. Synthetic task: irregular continuous-time system
# ============================================================
# Hidden mode z_t switches between multiple time constants.
# State update uses the exact solution of a 1st-order continuous-time system:
#
#   y_{t+1} = exp(-dt_t / tau(z_t)) * y_t
#             + (1 - exp(-dt_t / tau(z_t))) * gain(z_t) * u_t
#
# The model does NOT observe z_t directly. It only sees:
#   - the drive u_t
#   - a noisy observation of the current response x_t = y_t + noise
#   - the elapsed time dt_t
#
# This task is deliberately aligned with what LTC is good at:
#   - continuous-time dynamics
#   - irregular sampling
#   - switching time scales
# ============================================================

@dataclass
class TaskConfig:
    seq_len: int
    tau_values: List[float]
    gain_values: List[float]
    noise_std: float
    drive_change_prob: float
    impulse_prob: float
    switch_prob: float
    dt_small_min: float
    dt_small_max: float
    dt_large_min: float
    dt_large_max: float
    large_gap_prob: float


def generate_drive(seq_len: int, change_prob: float, impulse_prob: float) -> np.ndarray:
    u = np.zeros(seq_len, dtype=np.float32)
    current = 0.0
    for t in range(seq_len):
        if t == 0 or np.random.rand() < change_prob:
            # Mostly piecewise-constant bursts, occasionally silence.
            current = np.random.uniform(-1.6, 1.6) if np.random.rand() < 0.8 else 0.0
        impulse = np.random.uniform(-2.5, 2.5) if np.random.rand() < impulse_prob else 0.0
        u[t] = current + impulse
    return u


def generate_mode_sequence(seq_len: int, n_modes: int, switch_prob: float) -> np.ndarray:
    z = np.zeros(seq_len, dtype=np.int64)
    z[0] = np.random.randint(0, n_modes)
    for t in range(1, seq_len):
        if np.random.rand() < switch_prob:
            choices = [m for m in range(n_modes) if m != z[t - 1]]
            z[t] = np.random.choice(choices)
        else:
            z[t] = z[t - 1]
    return z


def generate_dt(seq_len: int, cfg: TaskConfig) -> Tuple[np.ndarray, np.ndarray]:
    dt = np.random.uniform(cfg.dt_small_min, cfg.dt_small_max, size=seq_len).astype(np.float32)
    large_gap_mask = np.random.rand(seq_len) < cfg.large_gap_prob
    if np.any(large_gap_mask):
        dt[large_gap_mask] = np.random.uniform(
            cfg.dt_large_min,
            cfg.dt_large_max,
            size=int(np.sum(large_gap_mask)),
        ).astype(np.float32)
    return dt, large_gap_mask.astype(np.float32)


def generate_sequence(cfg: TaskConfig) -> Dict[str, np.ndarray]:
    tau_values = np.asarray(cfg.tau_values, dtype=np.float32)
    gain_values = np.asarray(cfg.gain_values, dtype=np.float32)
    assert len(tau_values) == len(gain_values)

    seq_len = cfg.seq_len
    dt, large_gap_mask = generate_dt(seq_len, cfg)
    u = generate_drive(seq_len, cfg.drive_change_prob, cfg.impulse_prob)
    z = generate_mode_sequence(seq_len, len(tau_values), cfg.switch_prob)

    y = np.zeros(seq_len + 1, dtype=np.float32)
    for t in range(seq_len):
        tau = tau_values[z[t]]
        gain = gain_values[z[t]]
        alpha = np.exp(-dt[t] / tau).astype(np.float32)
        y[t + 1] = alpha * y[t] + (1.0 - alpha) * (gain * u[t])

    clean = y[1:]
    noisy_obs = clean + cfg.noise_std * np.random.randn(seq_len).astype(np.float32)

    switch_mask = np.zeros(seq_len, dtype=np.float32)
    switch_mask[1:] = (z[1:] != z[:-1]).astype(np.float32)
    true_tau = tau_values[z]

    # General models get dt as a normal feature.
    x_general = np.stack([u, noisy_obs, dt], axis=-1).astype(np.float32)
    # LTC gets the same information but uses dt via official timespans.
    x_ltc = np.stack([u, noisy_obs], axis=-1).astype(np.float32)

    time_axis = np.cumsum(dt).astype(np.float32)

    return {
        "x_general": x_general,
        "x_ltc": x_ltc,
        "dt": dt[:, None].astype(np.float32),
        "u": u[:, None].astype(np.float32),
        "clean": clean[:, None].astype(np.float32),
        "noisy": noisy_obs[:, None].astype(np.float32),
        "true_tau": true_tau[:, None].astype(np.float32),
        "switch_mask": switch_mask[:, None].astype(np.float32),
        "large_gap_mask": large_gap_mask[:, None].astype(np.float32),
        "time": time_axis[:, None].astype(np.float32),
    }


def build_window_dataset(
    num_sequences: int,
    history_len: int,
    stride: int,
    cfg: TaskConfig,
    switch_context: int,
) -> Dict[str, np.ndarray]:
    x_general_all = []
    x_ltc_all = []
    dt_all = []
    y_all = []
    hard_mask_all = []
    switch_target_all = []
    gap_target_all = []

    for _ in range(num_sequences):
        seq = generate_sequence(cfg)
        seq_len = cfg.seq_len
        for start in range(0, seq_len - history_len - 1, stride):
            target_idx = start + history_len

            x_general_all.append(seq["x_general"][start:target_idx])
            x_ltc_all.append(seq["x_ltc"][start:target_idx])
            dt_all.append(seq["dt"][start:target_idx])
            y_all.append(seq["clean"][target_idx])

            left = max(0, target_idx - switch_context)
            right = min(seq_len, target_idx + switch_context + 1)
            near_switch = float(np.any(seq["switch_mask"][left:right] > 0.5))
            large_gap = float(seq["large_gap_mask"][target_idx, 0] > 0.5)
            hard = float((near_switch > 0.5) or (large_gap > 0.5))

            switch_target_all.append([near_switch])
            gap_target_all.append([large_gap])
            hard_mask_all.append([hard])

    return {
        "x_general": np.stack(x_general_all).astype(np.float32),
        "x_ltc": np.stack(x_ltc_all).astype(np.float32),
        "dt": np.stack(dt_all).astype(np.float32),
        "y": np.stack(y_all).astype(np.float32),
        "hard_mask": np.stack(hard_mask_all).astype(np.float32),
        "switch_mask": np.stack(switch_target_all).astype(np.float32),
        "gap_mask": np.stack(gap_target_all).astype(np.float32),
    }


# ============================================================
# 3. Models
# ============================================================

class FNNPredictor(nn.Module):
    def __init__(self, history_len: int, input_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(history_len * input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_general: torch.Tensor, x_ltc: torch.Tensor = None, timespans: torch.Tensor = None) -> torch.Tensor:
        return self.net(x_general)


class RNNPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x_general: torch.Tensor, x_ltc: torch.Tensor = None, timespans: torch.Tensor = None) -> torch.Tensor:
        out, _ = self.rnn(x_general)
        return self.head(out[:, -1, :])


class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x_general: torch.Tensor, x_ltc: torch.Tensor = None, timespans: torch.Tensor = None) -> torch.Tensor:
        out, _ = self.lstm(x_general)
        return self.head(out[:, -1, :])


class LTCPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, ode_unfolds: int):
        super().__init__()
        self.ltc = LTC(
            input_size=input_size,
            units=hidden_size,
            batch_first=True,
            ode_unfolds=ode_unfolds,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x_general: torch.Tensor = None, x_ltc: torch.Tensor = None, timespans: torch.Tensor = None) -> torch.Tensor:
        # The public ncps LTC.forward squeezes the timespans tensor and can break
        # batched irregular sampling. We therefore step through the official LTCCell
        # directly, still using the official ncps implementation and parameters.
        batch_size, seq_len, _ = x_ltc.shape
        device = x_ltc.device
        h_state = torch.zeros((batch_size, self.ltc.state_size), device=device)
        outputs = []
        for t in range(seq_len):
            xt = x_ltc[:, t, :]
            elapsed = timespans[:, t, :] if timespans.dim() == 3 else timespans[:, t : t + 1]
            h_out, h_state = self.ltc.rnn_cell(xt, h_state, elapsed)
            outputs.append(h_out)
        out = torch.stack(outputs, dim=1)
        return self.head(out[:, -1, :])


# ============================================================
# 4. Fair parameter matching
# ============================================================

def build_model(model_name: str, history_len: int, hidden_size: int, ode_unfolds: int) -> nn.Module:
    if model_name == "FNN":
        return FNNPredictor(history_len=history_len, input_size=3, hidden_size=hidden_size)
    if model_name == "RNN":
        return RNNPredictor(input_size=3, hidden_size=hidden_size)
    if model_name == "LSTM":
        return LSTMPredictor(input_size=3, hidden_size=hidden_size)
    if model_name == "LTC":
        return LTCPredictor(input_size=2, hidden_size=hidden_size, ode_unfolds=ode_unfolds)
    raise ValueError(f"Unknown model_name: {model_name}")


def match_hidden_size(
    model_name: str,
    history_len: int,
    target_params: int,
    ode_unfolds: int,
    search_max_hidden: int = 96,
) -> Tuple[int, int]:
    best_gap = None
    best_h = None
    best_p = None
    for h in range(4, search_max_hidden + 1):
        model = build_model(model_name, history_len, h, ode_unfolds)
        p = count_parameters(model)
        gap = abs(p - target_params)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_h = h
            best_p = p
    return best_h, best_p


# ============================================================
# 5. Training / evaluation
# ============================================================

@dataclass
class TrainResult:
    model_name: str
    hidden_size: int
    n_params: int
    train_losses: List[float]
    id_mse: float
    stress_mse: float
    stress_hard_mse: float
    stress_switch_mse: float
    stress_gap_mse: float
    model: nn.Module


def make_loader(dataset: Dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(dataset["x_general"]),
            torch.tensor(dataset["x_ltc"]),
            torch.tensor(dataset["dt"]),
            torch.tensor(dataset["y"]),
            torch.tensor(dataset["hard_mask"]),
            torch.tensor(dataset["switch_mask"]),
            torch.tensor(dataset["gap_mask"]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def masked_mse(pred: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.reshape(-1) > 0.5
    if not np.any(mask):
        return float("nan")
    return float(np.mean((pred[mask] - tgt[mask]) ** 2))


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds = []
    tgts = []
    hard = []
    sw = []
    gap = []
    with torch.no_grad():
        for xg, xl, dt, y, hard_mask, switch_mask, gap_mask in loader:
            pred = model(
                x_general=xg.to(device),
                x_ltc=xl.to(device),
                timespans=dt.to(device),
            ).cpu().numpy()
            preds.append(pred)
            tgts.append(y.numpy())
            hard.append(hard_mask.numpy())
            sw.append(switch_mask.numpy())
            gap.append(gap_mask.numpy())

    pred = np.concatenate(preds, axis=0)
    tgt = np.concatenate(tgts, axis=0)
    hard_mask = np.concatenate(hard, axis=0)
    switch_mask = np.concatenate(sw, axis=0)
    gap_mask = np.concatenate(gap, axis=0)

    return {
        "mse": float(np.mean((pred - tgt) ** 2)),
        "hard_mse": masked_mse(pred, tgt, hard_mask),
        "switch_mse": masked_mse(pred, tgt, switch_mask),
        "gap_mse": masked_mse(pred, tgt, gap_mask),
    }


def train_model(
    model_name: str,
    hidden_size: int,
    history_len: int,
    ode_unfolds: int,
    train_loader: DataLoader,
    id_test_loader: DataLoader,
    stress_test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> TrainResult:
    model = build_model(model_name, history_len, hidden_size, ode_unfolds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xg, xl, dt, y, _, _, _ in train_loader:
            xg = xg.to(device)
            xl = xl.to(device)
            dt = dt.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x_general=xg, x_ltc=xl, timespans=dt)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = xg.size(0)
            total_loss += loss.item() * bs
            total_count += bs

        train_losses.append(total_loss / max(total_count, 1))

    id_metrics = evaluate_model(model, id_test_loader, device)
    stress_metrics = evaluate_model(model, stress_test_loader, device)

    return TrainResult(
        model_name=model_name,
        hidden_size=hidden_size,
        n_params=count_parameters(model),
        train_losses=train_losses,
        id_mse=id_metrics["mse"],
        stress_mse=stress_metrics["mse"],
        stress_hard_mse=stress_metrics["hard_mse"],
        stress_switch_mse=stress_metrics["switch_mse"],
        stress_gap_mse=stress_metrics["gap_mse"],
        model=model,
    )


# ============================================================
# 6. Official LTC effective tau extraction
# ============================================================

def extract_ltc_tau_trace(
    ltc_model: LTCPredictor,
    x_ltc_seq: np.ndarray,
    dt_seq: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Extract a time-varying effective tau from the trained official ncps LTCCell.

    Following LTCCell._ode_solver, the conductance denominator changes with inputs and state.
    A useful interpretation is:

        tau_eff(t) ~ cm / (gleak + sensory_conductance + recurrent_conductance)

    This is not a separate teaching model. It is computed from the official LTCCell internals.
    """
    ltc_model.eval()
    cell = ltc_model.ltc.rnn_cell

    x = torch.tensor(x_ltc_seq[None, ...], dtype=torch.float32, device=device)
    dt = torch.tensor(dt_seq.reshape(1, -1), dtype=torch.float32, device=device)

    hidden = torch.zeros((1, cell.state_size), device=device)
    tau_trace = []

    with torch.no_grad():
        for t in range(x.shape[1]):
            xt_raw = x[:, t, :]
            xt = cell._map_inputs(xt_raw)

            sensory_w = cell.make_positive_fn(cell._params["sensory_w"])
            sensory_act = sensory_w * cell._sigmoid(
                xt,
                cell._params["sensory_mu"],
                cell._params["sensory_sigma"],
            )
            sensory_act = sensory_act * cell._params["sensory_sparsity_mask"]
            sensory_cond = torch.sum(sensory_act, dim=1)

            rec_w = cell.make_positive_fn(cell._params["w"])
            rec_act = rec_w * cell._sigmoid(
                hidden,
                cell._params["mu"],
                cell._params["sigma"],
            )
            rec_act = rec_act * cell._params["sparsity_mask"]
            rec_cond = torch.sum(rec_act, dim=1)

            gleak = cell.make_positive_fn(cell._params["gleak"])
            cm = cell.make_positive_fn(cell._params["cm"])
            tau_eff = cm / (gleak + sensory_cond + rec_cond + cell._epsilon)
            tau_trace.append(tau_eff.squeeze(0).cpu().numpy())

            elapsed = dt[:, t].squeeze(0)
            _, hidden = cell(xt_raw, hidden, elapsed)

    return np.stack(tau_trace, axis=0)  # [T, H]


# ============================================================
# 7. Sequence-level qualitative predictions
# ============================================================

def predict_on_single_sequence(
    model: nn.Module,
    seq: Dict[str, np.ndarray],
    history_len: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_general = seq["x_general"]
    x_ltc = seq["x_ltc"]
    dt = seq["dt"]
    clean = seq["clean"]
    time_axis = seq["time"]

    xg_all = []
    xl_all = []
    dt_all = []
    target = []
    t_out = []
    for start in range(0, len(clean) - history_len - 1):
        end = start + history_len
        xg_all.append(x_general[start:end])
        xl_all.append(x_ltc[start:end])
        dt_all.append(dt[start:end])
        target.append(clean[end].squeeze())
        t_out.append(time_axis[end].squeeze())

    xg = torch.tensor(np.stack(xg_all).astype(np.float32), device=device)
    xl = torch.tensor(np.stack(xl_all).astype(np.float32), device=device)
    ts = torch.tensor(np.stack(dt_all).astype(np.float32), device=device)

    model.eval()
    with torch.no_grad():
        pred = model(x_general=xg, x_ltc=xl, timespans=ts).cpu().numpy().squeeze(-1)

    return np.asarray(t_out), np.asarray(target), pred


# ============================================================
# 8. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="mode_switch_demo_outputs")

    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--window_stride", type=int, default=3)
    parser.add_argument("--switch_context", type=int, default=3)

    parser.add_argument("--seq_len", type=int, default=140)
    parser.add_argument("--tau_values", type=str, default="0.03,0.18,1.20")
    parser.add_argument("--gain_values", type=str, default="1.20,0.90,0.55")
    parser.add_argument("--noise_std", type=float, default=0.05)

    parser.add_argument("--drive_change_prob", type=float, default=0.09)
    parser.add_argument("--impulse_prob", type=float, default=0.03)
    parser.add_argument("--switch_prob_train", type=float, default=0.06)
    parser.add_argument("--switch_prob_stress", type=float, default=0.12)

    parser.add_argument("--dt_small_min", type=float, default=0.02)
    parser.add_argument("--dt_small_max", type=float, default=0.14)
    parser.add_argument("--dt_large_min", type=float, default=0.28)
    parser.add_argument("--dt_large_max", type=float, default=0.80)
    parser.add_argument("--large_gap_prob_train", type=float, default=0.08)
    parser.add_argument("--large_gap_prob_stress", type=float, default=0.20)

    parser.add_argument("--train_sequences", type=int, default=120)
    parser.add_argument("--id_test_sequences", type=int, default=40)
    parser.add_argument("--stress_test_sequences", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--target_params", type=int, default=900)
    parser.add_argument("--ode_unfolds", type=int, default=4)
    parser.add_argument("--tau_neurons_to_plot", type=int, default=6)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau_values = parse_csv_floats(args.tau_values)
    gain_values = parse_csv_floats(args.gain_values)
    assert len(tau_values) == len(gain_values), "tau_values and gain_values must have the same length"

    train_cfg = TaskConfig(
        seq_len=args.seq_len,
        tau_values=tau_values,
        gain_values=gain_values,
        noise_std=args.noise_std,
        drive_change_prob=args.drive_change_prob,
        impulse_prob=args.impulse_prob,
        switch_prob=args.switch_prob_train,
        dt_small_min=args.dt_small_min,
        dt_small_max=args.dt_small_max,
        dt_large_min=args.dt_large_min,
        dt_large_max=args.dt_large_max,
        large_gap_prob=args.large_gap_prob_train,
    )
    stress_cfg = TaskConfig(
        seq_len=args.seq_len,
        tau_values=tau_values,
        gain_values=gain_values,
        noise_std=args.noise_std,
        drive_change_prob=args.drive_change_prob,
        impulse_prob=args.impulse_prob,
        switch_prob=args.switch_prob_stress,
        dt_small_min=args.dt_small_min,
        dt_small_max=args.dt_small_max,
        dt_large_min=args.dt_large_min,
        dt_large_max=args.dt_large_max,
        large_gap_prob=args.large_gap_prob_stress,
    )

    print("Building datasets...")
    train_data = build_window_dataset(
        num_sequences=args.train_sequences,
        history_len=args.history_len,
        stride=args.window_stride,
        cfg=train_cfg,
        switch_context=args.switch_context,
    )
    id_test_data = build_window_dataset(
        num_sequences=args.id_test_sequences,
        history_len=args.history_len,
        stride=args.window_stride,
        cfg=train_cfg,
        switch_context=args.switch_context,
    )
    stress_test_data = build_window_dataset(
        num_sequences=args.stress_test_sequences,
        history_len=args.history_len,
        stride=args.window_stride,
        cfg=stress_cfg,
        switch_context=args.switch_context,
    )

    train_loader = make_loader(train_data, batch_size=args.batch_size, shuffle=True)
    id_test_loader = make_loader(id_test_data, batch_size=args.batch_size, shuffle=False)
    stress_test_loader = make_loader(stress_test_data, batch_size=args.batch_size, shuffle=False)

    model_names = ["FNN", "RNN", "LSTM", "LTC"]
    hidden_choices: Dict[str, int] = {}
    param_counts: Dict[str, int] = {}

    print("Matching parameter counts...")
    for name in model_names:
        h, p = match_hidden_size(
            model_name=name,
            history_len=args.history_len,
            target_params=args.target_params,
            ode_unfolds=args.ode_unfolds,
        )
        hidden_choices[name] = h
        param_counts[name] = p
        print(f"  {name:>4s}: hidden={h:>3d}, params={p}")

    results: Dict[str, TrainResult] = {}
    for name in model_names:
        print(f"Training {name}...")
        result = train_model(
            model_name=name,
            hidden_size=hidden_choices[name],
            history_len=args.history_len,
            ode_unfolds=args.ode_unfolds,
            train_loader=train_loader,
            id_test_loader=id_test_loader,
            stress_test_loader=stress_test_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
        results[name] = result
        print(
            f"  -> ID MSE={result.id_mse:.5f} | Stress MSE={result.stress_mse:.5f} "
            f"| Stress hard-region MSE={result.stress_hard_mse:.5f}"
        )

    # Held-out stress sequence for qualitative plots.
    demo_seq = generate_sequence(stress_cfg)
    demo_preds = {}
    demo_time = None
    demo_target = None
    for name in model_names:
        t_out, target, pred = predict_on_single_sequence(
            results[name].model,
            demo_seq,
            history_len=args.history_len,
            device=device,
        )
        demo_preds[name] = pred
        demo_time = t_out
        demo_target = target

    # Official LTC effective tau trace on the full stress-test sequence.
    tau_eff = extract_ltc_tau_trace(
        ltc_model=results["LTC"].model,
        x_ltc_seq=demo_seq["x_ltc"],
        dt_seq=demo_seq["dt"].squeeze(-1),
        device=device,
    )

    # ========================================================
    # Plot 1: signal / predictions / metrics / tau
    # ========================================================
    fig = plt.figure(figsize=(15, 14))

    # (a) stress-test continuous-time sequence
    ax1 = plt.subplot(4, 1, 1)
    t_full = demo_seq["time"].squeeze(-1)
    ax1.plot(t_full, demo_seq["u"].squeeze(-1), label="Input drive u(t)", alpha=0.7)
    ax1.plot(t_full, demo_seq["noisy"].squeeze(-1), label="Noisy observation", alpha=0.6)
    ax1.plot(t_full, demo_seq["clean"].squeeze(-1), label="Clean system response", linewidth=2)
    ax1b = ax1.twinx()
    ax1b.plot(t_full, demo_seq["true_tau"].squeeze(-1), linestyle="--", label="True hidden tau", alpha=0.8)
    ax1.set_title("Mode-switching continuous-time task (stress-test sequence)")
    ax1.set_xlabel("Continuous time")
    ax1.set_ylabel("Signal")
    ax1b.set_ylabel("True tau")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True)

    # (b) rolling one-step predictions
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(demo_time, demo_target, label="Target", linewidth=2)
    for name in model_names:
        ax2.plot(demo_time, demo_preds[name], label=name, alpha=0.9)
    ax2.set_title("Rolling one-step predictions on one held-out stress-test sequence")
    ax2.set_xlabel("Continuous time")
    ax2.set_ylabel("Prediction")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    # (c) grouped metrics
    ax3 = plt.subplot(4, 1, 3)
    labels = [f"{n}\n({results[n].n_params} params)" for n in model_names]
    x = np.arange(len(model_names))
    width = 0.18
    ax3.bar(x - 1.5 * width, [results[n].id_mse for n in model_names], width, label="ID MSE")
    ax3.bar(x - 0.5 * width, [results[n].stress_mse for n in model_names], width, label="Stress MSE")
    ax3.bar(x + 0.5 * width, [results[n].stress_switch_mse for n in model_names], width, label="Stress switch-region MSE")
    ax3.bar(x + 1.5 * width, [results[n].stress_gap_mse for n in model_names], width, label="Stress large-gap MSE")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("MSE")
    ax3.set_title("Fair comparison under irregular sampling and switching time scales")
    ax3.legend(loc="upper left")
    ax3.grid(True, axis="y")

    # (d) true tau vs LTC effective tau dynamics
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(t_full, demo_seq["true_tau"].squeeze(-1), label="True hidden tau", linewidth=2, linestyle="--")
    k = min(args.tau_neurons_to_plot, tau_eff.shape[1])
    for i in range(k):
        ax4.plot(t_full, tau_eff[:, i], label=f"LTC neuron {i} effective tau", alpha=0.8)
    ax4.set_title("Official ncps LTC: effective tau dynamics extracted from LTCCell")
    ax4.set_xlabel("Continuous time")
    ax4.set_ylabel("Tau")
    ax4.legend(loc="upper right", ncol=2)
    ax4.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "comparison_and_tau.png"), dpi=160)
    plt.close(fig)

    # ========================================================
    # Plot 2: training curves
    # ========================================================
    fig = plt.figure(figsize=(10, 6))
    for name in model_names:
        plt.plot(results[name].train_losses, label=name)
    plt.title("Training losses")
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=160)
    plt.close(fig)

    # Summary JSON
    summary = {
        name: {
            "hidden_size": results[name].hidden_size,
            "params": results[name].n_params,
            "id_mse": results[name].id_mse,
            "stress_mse": results[name].stress_mse,
            "stress_hard_mse": results[name].stress_hard_mse,
            "stress_switch_mse": results[name].stress_switch_mse,
            "stress_gap_mse": results[name].stress_gap_mse,
        }
        for name in model_names
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done. Files written to:", args.output_dir)
    for filename in ["comparison_and_tau.png", "training_curves.png", "summary.json"]:
        print("  -", os.path.join(args.output_dir, filename))


if __name__ == "__main__":
    main()
