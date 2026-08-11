
import math
import time
from typing import Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# ============================================================
# Discrete torque wrapper
# ============================================================

class DiscreteTorqueWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_map = np.array([
            [-1.0, -1.0],
            [-1.0,  0.0],
            [-1.0,  1.0],
            [ 0.0, -1.0],
            [ 0.0,  0.0],
            [ 0.0,  1.0],
            [ 1.0, -1.0],
            [ 1.0,  0.0],
            [ 1.0,  1.0],
        ], dtype=np.float64)

        self.action_space = gym.spaces.Discrete(len(self.action_map))
        self.observation_space = env.observation_space

    def action(self, act_idx: int):
        a = self.action_map[int(act_idx)]
        tau_max = float(self.env.torque_limit)
        return tau_max * a

    def reverse_action(self, tau):
        tau = np.asarray(tau, dtype=np.float64)
        tau_max = max(1e-8, float(self.env.torque_limit))
        a = np.round(tau / tau_max).astype(int)
        for i, v in enumerate(self.action_map.astype(int)):
            if np.all(v == a):
                return i
        return 4

# ============================================================
# Tile coding
# ============================================================

class TileCoder:
    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        n_tilings: int = 16,
        tiles_per_dim: Tuple[int, ...] = (8, 8, 10, 10, 8, 8),
        memory_size: int = 2**15,
        seed: int = 0,
    ):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.n_tilings = int(n_tilings)
        self.tiles_per_dim = np.asarray(tiles_per_dim, dtype=np.int64)
        self.memory_size = int(memory_size)
        self.rng = np.random.default_rng(seed)

        assert self.low.shape == self.high.shape
        assert len(self.tiles_per_dim) == len(self.low)

        self.dim = len(self.low)
        span = np.maximum(self.high - self.low, 1e-8)
        self.scale = self.tiles_per_dim / span

        self.offsets = np.zeros((self.n_tilings, self.dim), dtype=np.float64)
        for t in range(self.n_tilings):
            self.offsets[t] = (t / self.n_tilings) * (1.0 / self.tiles_per_dim)

        self.hash_basis = [int(x) for x in self.rng.integers(
            low=1, high=2**31 - 1, size=(self.dim + 1,), dtype=np.int64
        )]

    def _stable_hash(self, coords: np.ndarray) -> int:
        h = 1469598103934665603
        fnv_prime = 1099511628211
        mask = (1 << 64) - 1
        for i, c in enumerate(coords.astype(np.int64)):
            h ^= (int(c) + int(self.hash_basis[i])) & mask
            h = (h * fnv_prime) & mask
        return h % self.memory_size

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, self.low, self.high)
        u_base = (x - self.low) * self.scale

        active = np.empty(self.n_tilings, dtype=np.int64)
        for t in range(self.n_tilings):
            u = u_base + self.offsets[t] * self.tiles_per_dim
            coords = np.floor(u).astype(np.int64)
            full = np.concatenate(([t], coords))
            active[t] = self._stable_hash(full)
        return active

# ============================================================
# Linear Q approximator
# ============================================================

class LinearTileQ:
    def __init__(self, n_actions: int, tile_coder: TileCoder):
        self.n_actions = int(n_actions)
        self.tc = tile_coder
        self.w = np.zeros((self.n_actions, self.tc.memory_size), dtype=np.float64)

    def q_values(self, s: np.ndarray) -> np.ndarray:
        idx = self.tc.encode(s)
        return self.w[:, idx].sum(axis=1)

    def q(self, s: np.ndarray, a: int) -> float:
        idx = self.tc.encode(s)
        return float(self.w[int(a), idx].sum())

    def update(self, s: np.ndarray, a: int, target: float, alpha: float, td_clip=None) -> float:
        a = int(a)
        idx = self.tc.encode(s)
        q_sa = self.w[a, idx].sum()
        td = float(target - q_sa)

        if td_clip is not None:
            td = float(np.clip(td, -td_clip, td_clip))

        self.w[a, idx] += alpha * td
        return td

# ============================================================
# Schedules
# ============================================================

def linear_decay(start: float, end: float, frac: float, progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    if progress >= frac:
        return end
    r = progress / max(frac, 1e-8)
    return start + r * (end - start)

# ============================================================
# SARSA Agent
# ============================================================

class NStepSarsaTileAgent:
    def __init__(self, n_actions: int, tile_coder: TileCoder):
        self.qfunc = LinearTileQ(n_actions=n_actions, tile_coder=tile_coder)

    @property
    def n_actions(self) -> int:
        return self.qfunc.n_actions

    def greedy_action(self, s: np.ndarray) -> int:
        qv = self.qfunc.q_values(s)
        m = np.max(qv)
        best = np.flatnonzero(np.isclose(qv, m))
        return int(np.random.choice(best))

    def epsilon_greedy_action(self, s: np.ndarray, epsilon: float) -> Tuple[int, np.ndarray]:
        qv = self.qfunc.q_values(s)
        if np.random.rand() < epsilon:
            a = int(np.random.randint(self.n_actions))
        else:
            m = np.max(qv)
            best = np.flatnonzero(np.isclose(qv, m))
            a = int(np.random.choice(best))
        return a, qv

    def action(self, s: np.ndarray, epsilon: float = 0.0) -> int:
        if epsilon <= 0.0:
            return self.greedy_action(s)
        a, _ = self.epsilon_greedy_action(s, epsilon)
        return a

    @staticmethod
    def moving_average(x: np.ndarray, w: int = 25) -> np.ndarray:
        if len(x) == 0:
            return x
        w = max(1, int(w))
        y = np.zeros_like(x, dtype=np.float64)
        csum = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
        for i in range(len(x)):
            lo = max(0, i - w + 1)
            n = i - lo + 1
            y[i] = (csum[i + 1] - csum[lo]) / n
        return y

    @staticmethod
    def _compute_episode_metrics(distances, taus, epsilon: float = 0.15):
        """
        Same definitions as SARSA logger (except TD):
          mean_tracking_error = mean(distance_t)
          success_rate = (1/T) * sum_t 1(distance_t < epsilon)
          control_energy = sum_t ||tau_t||^2
          smoothness = sum_t ||tau_t - tau_{t-1}||^2
        """
        e = np.asarray(distances, dtype=np.float64)
        tau = np.asarray(taus, dtype=np.float64)

        mean_tracking_error = float(np.mean(e)) if len(e) else np.nan
        success_rate = float(np.mean(e < epsilon)) if len(e) else 0.0
        control_energy = float(np.sum(np.sum(tau ** 2, axis=1))) if len(tau) else 0.0

        if len(tau) >= 2:
            dtau = tau[1:] - tau[:-1]
            smoothness = float(np.sum(np.sum(dtau ** 2, axis=1)))
        else:
            smoothness = 0.0

        return mean_tracking_error, success_rate, control_energy, smoothness

    def train(
        self,
        env: gym.Env,
        n_step: int = 5,
        gamma: float = 0.99,
        alpha: float = 0.1,
        n_episodes: int = 600,
        eps_start: float = 0.30,
        eps_end: float = 0.03,
        eps_decay_frac: float = 0.7,
        seed: Optional[int] = 0,
        verbose_every: int = 20,
    ) -> Dict[str, list]:
        """
        Returns ONLY:
          - ret
          - td_abs
          - mean_tracking_error
          - success_rate
          - control_energy
          - smoothness
        """
        if seed is not None:
            np.random.seed(seed)

        alpha_eff = alpha / self.qfunc.tc.n_tilings

        logs = {
            "ret": [],
            "td_abs": [],               # per-episode mean |TD|
            "mean_tracking_error": [],
            "success_rate": [],
            "control_energy": [],
            "smoothness": [],
        }

        for ep in range(n_episodes):
            t0_wall = time.time()
            progress = ep / max(1, n_episodes - 1)
            epsilon = linear_decay(eps_start, eps_end, eps_decay_frac, progress)

            s, info = env.reset(seed=None if seed is None else (seed + ep))
            a, _ = self.epsilon_greedy_action(s, epsilon)

            states = [s]
            actions = [a]
            rewards = [0.0]
            T = math.inf

            ep_return = 0.0
            distances = []
            taus = []
            td_abs_steps = []

            t = 0
            while True:
                if t < T:
                    s_next, r, terminated, truncated, info = env.step(actions[t])

                    ep_return += float(r)
                    rewards.append(float(r))
                    distances.append(float(info.get("distance", np.nan)))
                    taus.append(info["tau"].astype(np.float64))
                    states.append(s_next)

                    if terminated:
                        T = t + 1
                    else:
                        a_next, _ = self.epsilon_greedy_action(s_next, epsilon)
                        actions.append(a_next)
                        if truncated:
                            T = t + 1

                tau_idx = t - n_step + 1
                if tau_idx >= 0:
                    G = 0.0
                    upper = int(min(tau_idx + n_step, T)) if T < math.inf else (tau_idx + n_step)
                    for k in range(tau_idx + 1, upper + 1):
                        G += (gamma ** (k - tau_idx - 1)) * rewards[k]

                    should_bootstrap = ((tau_idx + n_step) < T)
                    if should_bootstrap:
                        G += (gamma ** n_step) * self.qfunc.q(states[tau_idx + n_step], actions[tau_idx + n_step])

                    td_clip = None
                    if len(logs["td_abs"]) > 500:
                        td_clip = 10#np.mean(logs["td_abs"][-10:]) * 1.1

                    td = self.qfunc.update(states[tau_idx], actions[tau_idx], G, alpha_eff, td_clip)
                    td_abs_steps.append(abs(td))

                if tau_idx == T - 1:
                    break
                t += 1

            mean_abs_td = float(np.mean(td_abs_steps)) if td_abs_steps else 0.0
            mean_tracking_error, success_rate, control_energy, smoothness = self._compute_episode_metrics(
                distances, taus, epsilon=0.15
            )

            logs["ret"].append(float(ep_return))
            logs["td_abs"].append(mean_abs_td)
            logs["mean_tracking_error"].append(mean_tracking_error)
            logs["success_rate"].append(success_rate)
            logs["control_energy"].append(control_energy)
            logs["smoothness"].append(smoothness)

            if verbose_every and ((ep + 1) % verbose_every == 0 or ep == 0):
                lo = max(0, ep - 19)

                ret20 = float(np.mean(logs["ret"][lo:ep+1]))
                track20 = float(np.mean(logs["mean_tracking_error"][lo:ep+1]))
                succR20 = float(np.mean(logs["success_rate"][lo:ep+1]))
                energy20 = float(np.mean(logs["control_energy"][lo:ep+1]))
                smooth20 = float(np.mean(logs["smoothness"][lo:ep+1]))
                td20 = float(np.mean(logs["td_abs"][lo:ep+1]))

                print(
                    f"[Ep {ep+1:4d}/{n_episodes}] "
                    f"ret={ep_return:8.3f} | ret20={ret20:8.3f} | "
                    f"track20={track20:6.3f} | succR20={succR20:5.2f} | "
                    f"energy20={energy20:8.2f} | smooth20={smooth20:8.2f} | "
                    f"eps={epsilon:5.3f} | |TD|20={td20:6.3f}"
                )

        return logs

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 50,
        epsilon_eval: float = 0.0,
        seed: Optional[int] = 1234,
    ) -> Dict[str, float]:
        returns = []
        lengths = []
        mean_tracking_errors = []
        success_rates = []
        control_energies = []
        smoothnesses = []
        final_distances = []
        mean_distances = []

        if seed is not None:
            np.random.seed(seed)

        best_seed = None
        best_return = -np.inf

        for ep in range(n_episodes):
            s, info = env.reset(seed=None if seed is None else (seed + ep))
            done = False

            ep_return = 0.0
            ep_len = 0
            dists = []
            taus = []

            while not done:
                if np.random.rand() < epsilon_eval:
                    a = np.random.randint(env.action_space.n)
                else:
                    a = self.greedy_action(s)

                s, r, terminated, truncated, info = env.step(a)
                done = bool(terminated or truncated)

                ep_return += float(r)
                ep_len += 1
                dists.append(float(info.get("distance", np.nan)))
                taus.append(info["tau"].astype(np.float64))

            mean_tracking_error, success_rate, control_energy, smoothness = self._compute_episode_metrics(
                dists, taus, epsilon=0.15
            )

            returns.append(ep_return)
            lengths.append(ep_len)
            mean_tracking_errors.append(mean_tracking_error)
            success_rates.append(success_rate)
            control_energies.append(control_energy)
            smoothnesses.append(smoothness)
            final_distances.append(float(info.get("distance", np.nan)))
            mean_distances.append(float(np.nanmean(dists)) if dists else np.nan)

            if ep_return > best_return:
                best_return = ep_return
                best_seed = seed + ep

        return {
            "episodes": n_episodes,
            "avg_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "avg_length": float(np.mean(lengths)),
            "avg_mean_tracking_error": float(np.nanmean(mean_tracking_errors)),
            "avg_success_rate": float(np.mean(success_rates)),
            "avg_control_energy": float(np.mean(control_energies)),
            "avg_smoothness": float(np.mean(smoothnesses)),
            "avg_final_distance": float(np.mean(final_distances)),
            "std_final_distance": float(np.std(final_distances)),
            "avg_mean_distance": float(np.mean(mean_distances)),
        }, best_seed


# ============================================================
# Minimal plotting for the requested lists
# ============================================================

def plot_logs_grid(
    logs: dict,
    name_map: dict = None,
    ma_window: int = 25,
    std_window: int = 25,
    title: str = "Training Metrics",
    figsize_scale: float = 4.2,
):
    """
    Plot all 1D numeric series in `logs` on a near-square subplot grid.

    Parameters
    ----------
    logs : dict
        Dict like {"ret": [...], "success_rate": [...], ...}
    name_map : dict, optional
        Mapping from log-key -> pretty plot title, e.g.
        {"ret": "Return", "mean_tracking_error": "Tracking Error"}
    ma_window : int
        Moving-average window.
    std_window : int
        Rolling std window (used for shaded band around MA).
    title : str
        Figure title.
    figsize_scale : float
        Controls subplot size; figure size becomes ~ (cols*scale, rows*scale).

    Notes
    -----
    - Automatically ignores non-1D or non-numeric entries.
    - Plots:
        raw curve (light)
        moving average (bold)
        MA ± rolling std shaded band
    """

    import math
    import numpy as np
    import matplotlib.pyplot as plt

    if name_map is None:
        name_map = {}

    def _to_1d_float_array(x):
        try:
            a = np.asarray(x, dtype=np.float64)
        except Exception:
            return None
        if a.ndim != 1 or a.size == 0:
            return None
        return a

    def _moving_average(x, w):
        w = max(1, int(w))
        y = np.zeros_like(x, dtype=np.float64)
        csum = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
        for i in range(len(x)):
            lo = max(0, i - w + 1)
            n = i - lo + 1
            y[i] = (csum[i + 1] - csum[lo]) / n
        return y

    def _rolling_std(x, w):
        w = max(1, int(w))
        out = np.zeros_like(x, dtype=np.float64)
        for i in range(len(x)):
            lo = max(0, i - w + 1)
            chunk = x[lo:i + 1]
            out[i] = float(np.nanstd(chunk))
        return out

    def _pretty_name(k):
        if k in name_map:
            return name_map[k]
        return k.replace("_", " ").strip().title()

    # Keep only plottable 1D numeric series
    series = []
    for k, v in logs.items():
        arr = _to_1d_float_array(v)
        if arr is None:
            continue
        if np.all(np.isnan(arr)):
            continue
        series.append((k, arr))

    if len(series) == 0:
        print("No plottable 1D numeric logs found.")
        return

    n = len(series)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(figsize_scale * cols, figsize_scale * rows))
    axes = np.atleast_1d(axes).ravel()

    # global style tweaks (no custom colors; use defaults)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    for ax, (k, y) in zip(axes, series):
        x = np.arange(1, len(y) + 1, dtype=np.int64)

        # Handle NaNs for MA/std while preserving shape
        y_clean = y.astype(np.float64).copy()
        # Replace isolated NaNs for smoothing computation using forward fill + zero fallback
        if np.any(np.isnan(y_clean)):
            isn = np.isnan(y_clean)
            if np.all(isn):
                y_clean[:] = 0.0
            else:
                # forward fill then backfill
                idx = np.where(~isn, np.arange(len(y_clean)), 0)
                np.maximum.accumulate(idx, out=idx)
                y_clean = y_clean[idx]
                first_valid = np.argmax(~isn)
                y_clean[:first_valid] = y[first_valid]

        ma = _moving_average(y_clean, ma_window)
        rs = _rolling_std(y_clean, std_window)

        # Raw series
        ax.plot(x, y, alpha=0.28, linewidth=1.2, label="Raw")

        # Shaded MA ± std
        lower = ma - rs
        upper = ma + rs
        ax.fill_between(x, lower, upper, alpha=0.18, label=f"MA±STD ({std_window})")

        # MA line
        ax.plot(x, ma, linewidth=2.2, label=f"MA ({ma_window})")

        # Formatting
        ax.set_title(_pretty_name(k), fontsize=11, pad=8)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)

        # Nice y-limits with a small padding if finite
        finite_mask = np.isfinite(y)
        if np.any(finite_mask):
            y_min = np.nanmin(y[finite_mask])
            y_max = np.nanmax(y[finite_mask])
            if np.isfinite(y_min) and np.isfinite(y_max):
                if y_max > y_min:
                    pad = 0.06 * (y_max - y_min)
                    ax.set_ylim(y_min - pad, y_max + pad)
                else:
                    pad = 1.0 if y_max == 0 else 0.1 * abs(y_max)
                    ax.set_ylim(y_min - pad, y_max + pad)

        # Add a compact stats box
        last_val = y[-1] if len(y) else np.nan
        txt = (
            f"last={last_val:.3f}\n"
            f"mean={np.nanmean(y):.3f}\n"
            f"std={np.nanstd(y):.3f}"
        )
        ax.text(
            0.98, 0.02, txt,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.12)
        )

        # Legend (small)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)

    # Hide unused axes
    for j in range(len(series), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

# ============================================================
# Train(Static Fixed Target)
# ============================================================
print(100 * "*")
print("Static Fixed Target!")

base_env = TwoLinkReacherEnv(
    dt=0.02,
    obs_mode="minimal",
    target_mode=static_target_fn,
    max_steps=150,
    torque_limit=2.0,
    reward_torque_penalty=0.0,
    render_mode=None,
    seed=42,
)

env = DiscreteTorqueWrapper(base_env)

obs_low = np.array([-2.0, -2.0, -np.pi, -np.pi, -8.0, -8.0], dtype=np.float64)
obs_high = np.array([ 2.0,  2.0,  np.pi,  np.pi,  8.0,  8.0], dtype=np.float64)

tc = TileCoder(
    low=obs_low,
    high=obs_high,
    n_tilings=8,
    tiles_per_dim=(8, 8, 10, 10, 8, 8),
    memory_size=2**16,
    seed=123,
)

sarsa_agent = NStepSarsaTileAgent(n_actions=env.action_space.n, tile_coder=tc)

sarsa_logs = sarsa_agent.train(
    env=env,
    n_step=15,
    gamma=0.95,
    alpha=0.17,
    n_episodes=300,
    eps_start=0.995,
    eps_end=0.03,
    eps_decay_frac=0.65,
    seed=7,
    verbose_every=25,
)

plot_logs_grid(sarsa_logs, name_map={
    "ret": "Episode Return",
    "td_abs": "Mean |TD|",
    "mean_tracking_error": "Mean Tracking Error",
    "success_rate": "Success Rate",
    "control_energy": "Control Energy",
    "smoothness": "Torque Smoothness",
})

sarsa_metrics, sarsa_best_seed = sarsa_agent.evaluate(
    env=env,
    n_episodes=100,
    epsilon_eval=0.0,
    seed=2025,
)

print("\nFinal Evaluation Metrics")
for k, v in sarsa_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

# ============================================================
# Train(Static Random Target)
# ============================================================
print(100 * "*")
print("Static Random Target!")

base_env = TwoLinkReacherEnv(
    dt=0.02,
    obs_mode="minimal",
    target_mode="static_random",
    max_steps=150,
    torque_limit=2.0,
    reward_torque_penalty=0.0,
    render_mode=None,
    seed=42,
)

env = DiscreteTorqueWrapper(base_env)

obs_low = np.array([-2.0, -2.0, -np.pi, -np.pi, -8.0, -8.0], dtype=np.float64)
obs_high = np.array([ 2.0,  2.0,  np.pi,  np.pi,  8.0,  8.0], dtype=np.float64)

tc = TileCoder(
    low=obs_low,
    high=obs_high,
    n_tilings=8,
    tiles_per_dim=(8, 8, 10, 10, 8, 8),
    memory_size=2**16,
    seed=123,
)

sarsa_agent = NStepSarsaTileAgent(n_actions=env.action_space.n, tile_coder=tc)

sarsa_logs = sarsa_agent.train(
    env=env,
    n_step=15,
    gamma=0.95,
    alpha=0.17,
    n_episodes=300,
    eps_start=0.995,
    eps_end=0.03,
    eps_decay_frac=0.65,
    seed=7,
    verbose_every=25,
)

plot_logs_grid(sarsa_logs, name_map={
    "ret": "Episode Return",
    "td_abs": "Mean |TD|",
    "mean_tracking_error": "Mean Tracking Error",
    "success_rate": "Success Rate",
    "control_energy": "Control Energy",
    "smoothness": "Torque Smoothness",
})

sarsa_metrics, sarsa_best_seed = sarsa_agent.evaluate(
    env=env,
    n_episodes=100,
    epsilon_eval=0.0,
    seed=2025,
)

print("\nFinal Evaluation Metrics")
for k, v in sarsa_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

# ============================================================
# Train(Circular Target)
# ============================================================
print(100 * "*")
print("Circular Moving Target!")

base_env = TwoLinkReacherEnv(
    dt=0.02,
    obs_mode="minimal",
    target_mode=circular_target_fn,
    max_steps=150,
    torque_limit=2.0,
    reward_torque_penalty=0.0,
    render_mode=None,
    seed=42,
)

env = DiscreteTorqueWrapper(base_env)

obs_low = np.array([-2.0, -2.0, -np.pi, -np.pi, -8.0, -8.0], dtype=np.float64)
obs_high = np.array([ 2.0,  2.0,  np.pi,  np.pi,  8.0,  8.0], dtype=np.float64)

tc = TileCoder(
    low=obs_low,
    high=obs_high,
    n_tilings=8,
    tiles_per_dim=(8, 8, 10, 10, 8, 8),
    memory_size=2**16,
    seed=123,
)

sarsa_agent = NStepSarsaTileAgent(n_actions=env.action_space.n, tile_coder=tc)

sarsa_logs = sarsa_agent.train(
    env=env,
    n_step=15,
    gamma=0.95,
    alpha=0.17,
    n_episodes=300,
    eps_start=0.995,
    eps_end=0.03,
    eps_decay_frac=0.65,
    seed=7,
    verbose_every=25,
)

plot_logs_grid(sarsa_logs, name_map={
    "ret": "Episode Return",
    "td_abs": "Mean |TD|",
    "mean_tracking_error": "Mean Tracking Error",
    "success_rate": "Success Rate",
    "control_energy": "Control Energy",
    "smoothness": "Torque Smoothness",
})

sarsa_metrics, sarsa_best_seed = sarsa_agent.evaluate(
    env=env,
    n_episodes=100,
    epsilon_eval=0.0,
    seed=2025,
)

print("\nFinal Evaluation Metrics")
for k, v in sarsa_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""* General description

  * This script trains and evaluates an on-policy n-step SARSA agent with linear function approximation using tile coding, for the 2-link reacher task.
  * Because the original environment uses continuous torques, it wraps the environment with a discrete-action wrapper that maps 9 discrete actions to combinations of joint torques in `{−τmax, 0, +τmax}` for each joint.
  * It trains separate agents (same algorithm/hyperparameters pattern) on three target scenarios: fixed static target, random static target per episode, and a moving circular target.
  * During training it logs return, TD magnitude, tracking error, success rate, control energy, and torque smoothness, and then plots these logs in a grid with moving-average + variability shading.

* Discrete torque wrapper

  * `DiscreteTorqueWrapper(env)`

    * A `gym.ActionWrapper` that converts a discrete action index into a 2D torque vector.
    * Defines `action_map` as the 9 Cartesian products of `{-1,0,+1}` for joint1/joint2, sets `action_space` to `Discrete(9)`, and passes through the original observation space.
  * `action(act_idx)`

    * Maps integer `act_idx` to an action vector `a ∈ {-1,0,+1}^2`, then scales by the environment’s torque limit `tau_max` to produce continuous torques `tau = tau_max * a`.
  * `reverse_action(tau)`

    * Attempts to map a continuous torque vector back to the nearest discrete action index by normalizing by `tau_max`, rounding to integers, and matching against `action_map`.
    * Returns the “do nothing” action (index `4`, corresponding to `[0,0]`) if no exact match is found.

* Tile coding

  * `TileCoder(low, high, n_tilings=16, tiles_per_dim=(...), memory_size=2**15, seed=0)`

    * Implements hashed tile coding to convert a continuous observation vector into a fixed set of active feature indices.
    * Uses `low/high` for clipping and normalization, multiple tilings with systematic offsets to improve resolution, and hashing into a fixed `memory_size` to keep the feature table bounded.
  * `_stable_hash(coords)`

    * Hashes an integer coordinate vector (including tiling id) into `[0, memory_size)` using an FNV-1a–style 64-bit rolling hash plus a random hash basis for decorrelation.
  * `encode(x)`

    * Clips `x` into `[low, high]`, scales to tile coordinates, applies each tiling’s offset, floors to integer coordinates, concatenates `[tiling_id, coords...]`, hashes, and returns the `n_tilings` active indices.

* Linear Q approximator (tile-coded)

  * `LinearTileQ(n_actions, tile_coder)`

    * Stores weights `w` with shape `(n_actions, memory_size)`. Each state activates `n_tilings` indices; `Q(s,a)` is the sum of weights at those indices for action `a`.
  * `q_values(s)`

    * Returns the vector `Q(s,·)` by summing active tile weights for each action.
  * `q(s, a)`

    * Returns scalar `Q(s,a)` for one action.
  * `update(s, a, target, alpha, td_clip=None)`

    * Computes TD error `td = target − Q(s,a)`, optionally clips it, then applies the semi-gradient update to the active tiles: `w[a, idx] += alpha * td`.
    * Returns the (possibly clipped) TD error for logging.

* Schedules

  * `linear_decay(start, end, frac, progress)`

    * Piecewise-linear schedule used for epsilon: interpolates from `start` to `end` over the first `frac` portion of training; then holds at `end`.

* SARSA agent (n-step, on-policy)

  * `NStepSarsaTileAgent(n_actions, tile_coder)`

    * Wraps `LinearTileQ` and provides epsilon-greedy action selection plus train/evaluate routines.
  * `n_actions` (property)

    * Exposes the number of discrete actions from the underlying Q function.
  * `greedy_action(s)`

    * Chooses an action among argmax actions of `Q(s,·)` with random tie-breaking.
  * `epsilon_greedy_action(s, epsilon)`

    * With probability `epsilon`, selects a random discrete action; otherwise selects a greedy action (with tie-breaking). Returns `(action, q_values)` for optional diagnostics.
  * `action(s, epsilon=0.0)`

    * Convenience selector: greedy if `epsilon<=0`, else epsilon-greedy.
  * `moving_average(x, w=25)`

    * Computes a simple trailing moving average (used as a utility; plotting also implements its own MA).
  * `compute_episode_metrics(errors, taus, epsilon=0.15)`

    * Computes per-episode metrics from logged distance errors and torque vectors:

      * mean tracking error = mean distance
      * success rate = fraction of steps with distance < `epsilon`
      * control energy = sum over time of `||tau||²`
      * smoothness = sum over time of `||Δtau||²`
  * `train(env, n_step=5, gamma=0.99, alpha=0.1, n_episodes=600, eps_start=0.30, eps_end=0.03, eps_decay_frac=0.7, seed=0, verbose_every=20)`

    * Implements on-policy n-step SARSA with tile-coded linear approximation. Key mechanics:

      * Effective step size is normalized by number of tilings: `alpha_eff = alpha / n_tilings`.
      * For each episode:

        * Resets env, picks initial action epsilon-greedily.
        * Collects sequences of `states`, `actions`, `rewards` until termination/truncation.
        * For each time `t`, computes the n-step return `G` for the update index `tau_idx = t-n_step+1`:

          * `G = sum_{k=tau_idx+1}^{min(tau_idx+n_step, T)} gamma^{k-tau_idx-1} r_k`
          * If bootstrapping is valid, adds `gamma^n_step * Q(s_{tau_idx+n_step}, a_{tau_idx+n_step})`.
        * Applies a semi-gradient update via `LinearTileQ.update(...)`.
      * Logs per-episode: total return, mean |TD|, and the four tracking/control metrics derived from env `info` (`distance`, `tau`).
      * Prints rolling 20-episode averages every `verbose_every` episodes.
      * Contains an optional TD clipping heuristic after many episodes (`td_clip`) to limit large updates.
    * Returns a `logs` dict containing only the requested lists: `ret`, `td_abs`, `mean_tracking_error`, `success_rate`, `control_energy`, `smoothness`.
  * `evaluate(env, n_episodes=50, epsilon_eval=0.0, seed=1234)`

    * Runs evaluation rollouts, mostly greedy (unless `epsilon_eval>0`), and returns summary statistics: mean/std return, mean length, mean success flag, and distance statistics.
    * Also tracks and returns `best_seed`, the episode seed that produced the highest return (useful for replaying a “best” episode).

* Plotting utility

  * `plot_logs_grid(logs, name_map=None, ma_window=25, std_window=25, title="Training Metrics", figsize_scale=4.2)`

    * Finds plottable 1D numeric series in `logs` and plots them in a near-square grid of subplots.
    * For each metric: plots the raw curve, a moving average, and a shaded band of MA ± rolling standard deviation; adds a small stats box (last/mean/std) and a compact legend.
    * Ignores non-1D or all-NaN entries and hides unused subplot axes.

* Training runs (three scenarios)

  * Static fixed target

    * Creates `TwoLinkReacherEnv` with `target_mode=static_target_fn`, wraps with `DiscreteTorqueWrapper`, defines observation bounds for tile coding, trains an `NStepSarsaTileAgent`, plots training logs, then evaluates over 100 episodes with greedy policy.
  * Static random target

    * Same pipeline, but `target_mode="static_random"` so each episode samples a new constant target.
  * Circular moving target

    * Same pipeline, but `target_mode=circular_target_fn` so the target moves continuously on a circle during the episode.
  * Common elements across all three blocks

    * Observation bounds (`obs_low/obs_high`) are manually specified for `(xt, yt, th1, th2, dth1, dth2)` corresponding to `obs_mode="minimal"`.
    * Tile-coder configuration is consistent (8 tilings, specified tiles-per-dimension, hashed memory).
    * SARSA hyperparameters are the same pattern across runs (n-step, gamma, alpha, epsilon schedule, episode count), enabling apples-to-apples comparison across target regimes.
"""

import os
from typing import Dict, Any, Tuple

import numpy as np
from IPython.display import Video, display
from gymnasium.wrappers import RecordVideo

# ============================================================
# SARSA: record one episode from a given seed
# ============================================================
def record_sarsa_episode_video_recordvideo(
    agent,
    env_kwargs: Dict[str, Any],
    best_seed: int,
    video_folder: str = "videos",
    name_prefix: str = "sarsa_best_episode",
    success_epsilon: float = 0.15,
) -> Tuple[str, Dict[str, float]]:
    """
    Records ONE greedy SARSA episode using gymnasium RecordVideo, prints metrics,
    and embeds the generated mp4 in Kaggle/Jupyter.

    Returns
    -------
    (video_path, metrics_dict)
    """
    os.makedirs(video_folder, exist_ok=True)

    # Force rgb_array (required by RecordVideo)
    env_kwargs = dict(env_kwargs)
    env_kwargs["render_mode"] = "rgb_array"

    base_env = TwoLinkReacherEnv(**env_kwargs)
    env = DiscreteTorqueWrapper(base_env)

    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: True,
        name_prefix=name_prefix,
    )

    obs, info = env.reset(seed=int(best_seed))
    done = False

    ep_return = 0.0
    ep_len = 0
    distances = []
    taus = []

    while not done:
        a = agent.action(obs, epsilon=0.0)  # greedy
        obs, reward, terminated, truncated, info = env.step(a)

        ep_return += float(reward)
        ep_len += 1
        distances.append(float(info.get("distance", np.nan)))
        if "tau" in info:
            taus.append(np.asarray(info["tau"], dtype=np.float64))

        done = bool(terminated or truncated)

    env.close()

    metrics = compute_metrics(distances, taus, epsilon=success_epsilon)
    metrics = {
        "seed": int(best_seed),
        "return": float(ep_return),
        "episode_length": int(ep_len),
        **metrics,
    }

    print("\nSARSA Episode Metrics (single recorded episode)")
    for k, v in metrics.items():
        print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

    # Find generated video file(s)
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4") and f.startswith(name_prefix)])
    if not video_files:
        raise RuntimeError(f"No video files found in {video_folder}")

    video_path = os.path.join(video_folder, video_files[-1])
    print(f"\nDisplaying: {video_path}")
    display(Video(video_path, embed=True))

    return video_path, metrics

# SARSA env kwargs (same family as training)
sarsa_env_kwargs = dict(
    dt=0.02,
    obs_mode="minimal",
    target_mode=circular_target_fn,
    max_steps=150,
    torque_limit=2.0,
    reward_torque_penalty=0.0,
    render_mode=None,  # overridden to rgb_array inside function
    seed=42,
)
sarsa_video_path, sarsa_ep_metrics = record_sarsa_episode_video_recordvideo(
    agent=sarsa_agent,
    env_kwargs=sarsa_env_kwargs,
    best_seed=sarsa_best_seed,
    video_folder="videos",
    name_prefix="sarsa_best",
)

"""# Part (e): Continuous Control with DDPG"""

import random
from dataclasses import dataclass
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=int(1e5)):
        self.capacity = int(capacity)
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, done):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.state[idx]),
            torch.tensor(self.action[idx]),
            torch.tensor(self.reward[idx]),
            torch.tensor(self.next_state[idx]),
            torch.tensor(self.done[idx]),
        )


# ============================================================
# Noise (exploration)
# ============================================================
class DDPGNoise:
    """
    Noise generator for DDPG exploration.

    Supported noise types:
        - 'ou'       : Ornstein-Uhlenbeck process (temporally correlated)
        - 'gaussian' : Uncorrelated Gaussian noise
    """
    def __init__(self, action_dim, noise_type='ou', **kwargs):
        self.action_dim = action_dim
        self.noise_type = noise_type.lower()

        if self.noise_type == 'ou':
            self.mu = kwargs.get('mu', 0.0)
            self.theta = kwargs.get('theta', 0.15)
            self.sigma = kwargs.get('sigma', 0.1)
            self.x = np.zeros(action_dim, dtype=np.float32)
            self.dt = kwargs.get('dt', 1)

        elif self.noise_type == 'gaussian':
            self.sigma = kwargs.get('sigma', 0.1)

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    def reset(self):
        """Reset the noise process to its initial state."""
        if self.noise_type == 'ou':
            self.x = np.zeros(self.action_dim, dtype=np.float32)
        # Gaussian noise has no state, so nothing to reset

    def sample(self):
        """
        Generate a noise sample of dimension `action_dim`.

        Returns
        -------
        np.ndarray
            Noise vector (float32).
        """
        if self.noise_type == 'ou':
            dx = self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
            self.x = self.x + dx
            return self.x.astype(np.float32)

        elif self.noise_type == 'gaussian':
            return (self.sigma * np.random.randn(self.action_dim)).astype(np.float32)


# ============================================================
# Networks
# ============================================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit):
        super().__init__()
        self.action_limit = float(action_limit)

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),  # scale to [-1,1]
        )

    def forward(self, s):
        return self.action_limit * self.net(s)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)