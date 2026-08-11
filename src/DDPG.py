
# ============================================================
# DDPG Agent
# ============================================================

@dataclass
class DDPGConfig:
    gamma: float = 0.99
    tau: float = 0.005              # soft-update coeff
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 128
    buffer_capacity: int = 100_000
    warmup_steps: int = 2000
    train_every: int = 1
    gradient_steps: int = 1
    ou_theta: float = 0.15
    ou_sigma: float = 0.20
    noise_type: str = "ou"
    dt: float = 1.0

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_limit, cfg: DDPGConfig, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.actor = Actor(state_dim, action_dim, action_limit).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        self.actor_target = Actor(state_dim, action_dim, action_limit).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.replay = ReplayBuffer(state_dim, action_dim, capacity=cfg.buffer_capacity)
        self.noise = DDPGNoise(action_dim, noise_type=self.cfg.noise_type, mu=0.0, theta=cfg.ou_theta, sigma=cfg.ou_sigma, dt=self.cfg.dt)

        self.action_limit = float(action_limit)
        self.total_steps = 0

    @torch.no_grad()
    def act(self, s, add_noise=True):
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s_t).cpu().numpy()[0]

        if add_noise:
            a = a + self.noise.sample()

        a = np.clip(a, -self.action_limit, self.action_limit)
        return a.astype(np.float32)

    def update(self):
        if self.replay.size < self.cfg.batch_size:
            return None

        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        # Critic target
        with torch.no_grad():
            a2 = self.actor_target(s2)
            q2 = self.critic_target(s2, a2)
            y = r + self.cfg.gamma * (1.0 - d) * q2

        # Critic loss
        q = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Actor loss (maximize Q => minimize -Q)
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Soft update
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def soft_update(self, net, target_net):
        tau = self.cfg.tau
        for p, p_targ in zip(net.parameters(), target_net.parameters()):
            p_targ.data.copy_(tau * p.data + (1.0 - tau) * p_targ.data)

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
        env,
        n_episodes: int = 300,
        seed: Optional[int] = 0,
        verbose_every: int = 25,
    ) -> Dict[str, list]:
        """
        Prints and returns the same per-episode lists as the SARSA logger,
        except td_abs (not used here).

        Returns dict with ONLY:
          - ret
          - mean_tracking_error
          - success_rate
          - control_energy
          - smoothness
        """
        if seed is not None:
            set_seed(seed)

        logs = {
            "ret": [],
            "mean_tracking_error": [],
            "success_rate": [],
            "control_energy": [],
            "smoothness": [],
        }

        for ep in range(n_episodes):
            t0_wall = time.time()

            s, info = env.reset(seed=None if seed is None else (seed + ep))
            self.noise.reset()

            ep_return = 0.0
            distances = []
            taus = []

            done = False
            ep_steps = 0

            while not done:
                # Warmup random actions
                if self.total_steps < self.cfg.warmup_steps:
                    a = np.random.uniform(
                        low=-self.action_limit,
                        high=self.action_limit,
                        size=(env.action_space.shape[0],),
                    ).astype(np.float32)
                else:
                    a = self.act(s, add_noise=True)

                s2, r, terminated, truncated, info = env.step(a)
                done = bool(terminated or truncated)

                # For Bellman targets, only true terminal states should cut bootstrap.
                # Time-limit truncation is not a terminal transition.
                done_for_bootstrap = float(terminated)

                ep_return += float(r)
                distances.append(float(info.get("distance", np.nan)))
                taus.append(info["tau"].astype(np.float64))

                # store transition
                self.replay.add(
                    s.astype(np.float32),
                    a.astype(np.float32),
                    float(r),
                    s2.astype(np.float32),
                    float(done_for_bootstrap),
                )

                s = s2
                ep_steps += 1
                self.total_steps += 1

                # updates (respect cfg.train_every and cfg.gradient_steps)
                if (self.total_steps >= self.cfg.warmup_steps) and (self.total_steps % self.cfg.train_every == 0):
                    for _ in range(self.cfg.gradient_steps):
                        self.update()

            # per-episode metrics (same as SARSA)
            mean_tracking_error, success_rate, control_energy, smoothness = self._compute_episode_metrics(
                distances, taus, epsilon=0.15
            )

            logs["ret"].append(float(ep_return))
            logs["mean_tracking_error"].append(mean_tracking_error)
            logs["success_rate"].append(success_rate)
            logs["control_energy"].append(control_energy)
            logs["smoothness"].append(smoothness)

            if verbose_every and ((ep + 1) % verbose_every == 0 or ep == 0):
                lo = max(0, ep - 19)

                ret20 = float(np.mean(logs["ret"][lo:ep + 1]))
                track20 = float(np.mean(logs["mean_tracking_error"][lo:ep + 1]))
                succR20 = float(np.mean(logs["success_rate"][lo:ep + 1]))
                energy20 = float(np.mean(logs["control_energy"][lo:ep + 1]))
                smooth20 = float(np.mean(logs["smoothness"][lo:ep + 1]))

                print(
                    f"[Ep {ep+1:4d}/{n_episodes}] "
                    f"ret={ep_return:8.3f} | ret20={ret20:8.3f} | "
                    f"track20={track20:6.3f} | succR20={succR20:5.2f} | "
                    f"energy20={energy20:8.2f} | smooth20={smooth20:8.2f}"
                )

        return logs

    def evaluate(
        self,
        env,
        n_episodes: int = 100,
        seed: Optional[int] = 2025,
    ) -> Dict[str, float]:
        """
        Greedy / deterministic evaluation (no exploration noise),
        reporting the same metrics (except td_abs).
        """
        if seed is not None:
            set_seed(seed)

        returns = []
        lengths = []
        mean_tracking_errors = []
        success_rates = []
        control_energies = []
        smoothnesses = []
        final_distances = []
        mean_distances = []

        best_seed = None
        best_return = -np.inf

        for ep in range(n_episodes):
            s, info = env.reset(seed=None if seed is None else (seed + ep))
            self.noise.reset()

            ep_return = 0.0
            ep_len = 0
            distances = []
            taus = []

            done = False
            while not done:
                a = self.act(s, add_noise=False)
                s, r, terminated, truncated, info = env.step(a)
                done = bool(terminated or truncated)

                ep_return += float(r)
                ep_len += 1
                distances.append(float(info.get("distance", np.nan)))
                taus.append(info["tau"].astype(np.float64))

            mean_tracking_error, success_rate, control_energy, smoothness = self._compute_episode_metrics(
                distances, taus, epsilon=0.15
            )

            returns.append(ep_return)
            lengths.append(ep_len)
            mean_tracking_errors.append(mean_tracking_error)
            success_rates.append(success_rate)
            control_energies.append(control_energy)
            smoothnesses.append(smoothness)
            final_distances.append(float(info.get("distance", np.nan)))
            mean_distances.append(float(np.nanmean(distances)) if distances else np.nan)

        if ep_return > best_return:
            best_return = ep_return
            best_seed = seed + ep

        metrics = {
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
        }
        return metrics, best_seed

# ============================================================
# Train(Static Fixed Target)
# ============================================================
print(100 * "*")
print("Static Fixed Target!")

cfg = DDPGConfig(
    gamma=0.95,
    tau=0.01,
    actor_lr=3e-4,
    critic_lr=1e-3,
    batch_size=64,
    buffer_capacity=100_000,
    warmup_steps=2000,
    train_every=1,
    gradient_steps=2,
    ou_theta=0.10,
    ou_sigma=0.15,
)

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode=static_target_fn,
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ddpg_agent = DDPGAgent(state_dim, action_dim, 2.0, cfg, device="cuda")
# train
ddpg_logs = ddpg_agent.train(env, n_episodes=300, seed=7, verbose_every=25)

plot_logs_grid(ddpg_logs, name_map={
    "ret": "Episode Return",
    "mean_tracking_error": "Mean Tracking Error",
    "success_rate": "Success Rate",
    "control_energy": "Control Energy",
    "smoothness": "Torque Smoothness",
}, title="DDPG Training Metrics")

# evaluate
ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=2025)

print("\nFinal Evaluation Metrics")
for k, v in ddpg_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

# ============================================================
# Train(Static Random Target)
# ============================================================
print(100 * "*")
print("Static Random Target!")

cfg = DDPGConfig(
    gamma=0.95,
    tau=0.01,
    actor_lr=3e-4,
    critic_lr=1e-3,
    batch_size=64,
    buffer_capacity=100_000,
    warmup_steps=2000,
    train_every=1,
    gradient_steps=2,
    ou_theta=0.10,
    ou_sigma=0.15,
)

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode="static_random",
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ddpg_agent = DDPGAgent(state_dim, action_dim, 2.0, cfg, device="cuda")
# train
ddpg_logs = ddpg_agent.train(env, n_episodes=300, seed=7, verbose_every=25)

plot_logs_grid(ddpg_logs, name_map={
    "ret": "Episode Return",
    "mean_tracking_error": "Mean Tracking Error",
    "success_rate": "Success Rate",
    "control_energy": "Control Energy",
    "smoothness": "Torque Smoothness",
}, title="DDPG Training Metrics")

# evaluate
ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=2025)

print("\nFinal Evaluation Metrics")
for k, v in ddpg_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

# ============================================================
# Train(Circular Moving Target)
# ============================================================
print(100 * "*")
print("Circular Moving Target!")

cfg = DDPGConfig(
    gamma=0.95,
    tau=0.01,
    actor_lr=3e-4,
    critic_lr=1e-3,
    batch_size=64,
    buffer_capacity=100_000,
    warmup_steps=2000,
    train_every=1,
    gradient_steps=2,
    ou_theta=0.10,
    ou_sigma=0.15,
)

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode=circular_target_fn,
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ddpg_agent = DDPGAgent(state_dim, action_dim, 2.0, cfg, device="cuda")
# train
ddpg_logs = ddpg_agent.train(env, n_episodes=300, seed=7, verbose_every=25)

plot_logs_grid(ddpg_logs, name_map={
    "ret": "Episode Return",
    "mean_tracking_error": "Mean Tracking Error",
    "success_rate": "Success Rate",
    "control_energy": "Control Energy",
    "smoothness": "Torque Smoothness",
}, title="DDPG Training Metrics")

# evaluate
ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=2025)

print("\nFinal Evaluation Metrics")
for k, v in ddpg_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""* General description

  * This script implements a basic Deep Deterministic Policy Gradient (DDPG) agent in PyTorch for continuous-control of the 2-link reacher environment.
  * It includes a replay buffer, configurable exploration noise (Ornstein–Uhlenbeck or Gaussian), actor/critic neural networks with corresponding target networks, and the standard DDPG training loop (off-policy updates with soft target updates).
  * It trains and evaluates the agent on three target regimes (static fixed, static random per episode, circular moving target), logging per-episode performance metrics compatible with the earlier SARSA logger (return, tracking error, success rate, control energy, smoothness), and plots training curves using `plot_logs_grid`.

* Utilities

  * `set_seed(seed=0)`

    * Sets Python `random`, NumPy, and PyTorch RNG seeds for reproducibility of sampling, initialization, and training behavior.

* Replay buffer

  * `ReplayBuffer(state_dim, action_dim, capacity=1e5)`

    * Fixed-size circular buffer storing transitions `(s, a, r, s2, done)` in preallocated numpy arrays for efficiency.
    * `done` is stored as a float mask used to stop bootstrapping for terminal transitions.
  * `add(s, a, r, s2, done)`

    * Inserts one transition at the current pointer, advances the pointer modulo capacity, and updates the current size.
  * `sample(batch_size)`

    * Uniformly samples indices from stored transitions and returns PyTorch tensors for `(state, action, reward, next_state, done)`.

* Exploration noise

  * `DDPGNoise(action_dim, noise_type='ou', **kwargs)`

    * Generates additive exploration noise for continuous actions. Supports:

      * `'ou'`: Ornstein–Uhlenbeck process (stateful, temporally correlated noise; common in classic DDPG)
      * `'gaussian'`: i.i.d. Gaussian noise (stateless)
  * `reset()`

    * Resets OU internal state to zeros at the start of an episode; Gaussian has no internal state.
  * `sample()`

    * Returns one noise vector (float32).
    * OU: updates internal state `x` via mean reversion + diffusion; Gaussian: returns `sigma * N(0, I)`.

* Networks

  * `Actor(state_dim, action_dim, action_limit)`

    * MLP policy mapping state → action. Uses `tanh` output scaled by `action_limit` so actions lie in `[-action_limit, +action_limit]`.
  * `forward(s)`

    * Produces deterministic action vector for the given state batch.
  * `Critic(state_dim, action_dim)`

    * MLP action-value function mapping concatenated `[state, action]` → scalar Q-value.
  * `forward(s, a)`

    * Concatenates state and action along the last dimension and outputs `Q(s,a)`.

* Configuration

  * `DDPGConfig` (dataclass)

    * Hyperparameters controlling discounting (`gamma`), target-network soft update (`tau`), learning rates, batch size, replay capacity, warmup steps before learning, update frequency, number of gradient steps per environment step, and exploration noise parameters/type.

* `DDPGAgent`

  * `__init__(state_dim, action_dim, action_limit, cfg, device="cuda")`

    * Builds actor/critic networks and their target copies, syncs target parameters, creates Adam optimizers, initializes replay buffer and noise process, and stores action bounds and a global step counter.
  * `act(s, add_noise=True)`

    * Computes deterministic actor output for a single state, optionally adds exploration noise, and clips the resulting action to `[-action_limit, +action_limit]`.
  * `update()`

    * Performs one DDPG update if enough replay samples exist:

      * Target: `y = r + gamma * (1-done) * Q_target(s2, actor_target(s2))`
      * Critic update: minimize MSE between `Q(s,a)` and `y`.
      * Actor update: maximize expected Q by minimizing `-Q(s, actor(s))`.
      * Applies gradient clipping (norm 1.0) and then soft-updates both target networks.
    * Returns a dict with `critic_loss` and `actor_loss` (or `None` if insufficient buffer).
  * `soft_update(net, target_net)`

    * Polyak averaging: `θ_target ← τ θ + (1−τ) θ_target` for all parameters.
  * `_compute_episode_metrics(distances, taus, epsilon=0.15)`

    * Computes the same episode-level metrics used in the SARSA experiments (mean tracking error, success rate by distance threshold, control energy, and torque smoothness).
  * `train(env, n_episodes=300, seed=0, verbose_every=25)`

    * Runs the DDPG interaction + learning loop and returns per-episode logs (no TD metric):

      * Resets env and noise each episode.
      * Chooses actions randomly during warmup (`warmup_steps`) to fill replay; afterward uses actor + noise.
      * Stores transitions in replay; uses `done_for_bootstrap = terminated` so time-limit truncations do not cut bootstrapping targets.
      * After warmup, performs updates every `train_every` steps, doing `gradient_steps` updates each time.
      * Logs `ret`, `mean_tracking_error`, `success_rate`, `control_energy`, `smoothness` and prints rolling 20-episode averages periodically.
  * `evaluate(env, n_episodes=100, seed=2025)`

    * Runs deterministic policy evaluation (no exploration noise) for multiple episodes and aggregates mean/std return and averaged metrics.
    * Returns `(metrics_dict, best_seed)` intended to identify a reproducible high-return episode seed.
    * Note: in the provided code, the `best_seed/best_return` update is placed outside the evaluation loop, so it effectively only considers the final episode’s return; logically it should be inside the loop to track the true best episode.

* Training runs (three scenarios)

  * Static fixed target

    * Creates `TwoLinkReacherEnv` with `target_mode=static_target_fn` and `obs_mode="minimal_ee"` (state includes end-effector position), constructs a `DDPGAgent`, trains for 300 episodes, plots logs, then evaluates for 100 episodes.
  * Static random target

    * Same pipeline but `target_mode="static_random"` so each episode samples a new fixed target.
  * Circular moving target

    * Same pipeline but `target_mode=circular_target_fn` so the target moves continuously along a circle.
  * Common elements across blocks

    * Uses the same `DDPGConfig` structure (gamma/tau/lrs/batch size/warmup/update schedule/noise params).
    * Logs and plots the same metric keys via `plot_logs_grid` for consistent comparison across target regimes.
"""

# ============================================================
# DDPG: record one episode from a given seed
# ============================================================
def record_ddpg_episode_video_recordvideo(
    agent,
    env_kwargs: Dict[str, Any],
    best_seed: int,
    video_folder: str = "videos",
    name_prefix: str = "ddpg_best_episode",
    success_epsilon: float = 0.15,
) -> Tuple[str, Dict[str, float]]:
    """
    Records ONE deterministic DDPG episode using gymnasium RecordVideo, prints metrics,
    and embeds the generated mp4 in Kaggle/Jupyter.

    Returns
    -------
    (video_path, metrics_dict)
    """
    os.makedirs(video_folder, exist_ok=True)

    env_kwargs = dict(env_kwargs)
    env_kwargs["render_mode"] = "rgb_array"

    env = TwoLinkReacherEnv(**env_kwargs)

    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: True,
        name_prefix=name_prefix,
    )

    obs, info = env.reset(seed=int(best_seed))
    if hasattr(agent, "noise"):
        agent.noise.reset()

    done = False
    ep_return = 0.0
    ep_len = 0
    distances = []
    taus = []

    while not done:
        a = agent.act(obs, add_noise=False)  # deterministic policy
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

    print("\nDDPG Episode Metrics (single recorded episode)")
    for k, v in metrics.items():
        print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4") and f.startswith(name_prefix)])
    if not video_files:
        raise RuntimeError(f"No video files found in {video_folder}")

    video_path = os.path.join(video_folder, video_files[-1])
    print(f"\nDisplaying: {video_path}")
    display(Video(video_path, embed=True))

    return video_path, metrics

ddpg_env_kwargs = dict(
    dt=0.02,
    obs_mode="minimal_ee",
    target_mode=circular_target_fn,
    max_steps=150,
    torque_limit=2.0,
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    render_mode=None,  # function overrides to rgb_array
)

ddpg_video_path, ddpg_ep_metrics = record_ddpg_episode_video_recordvideo(
    agent=ddpg_agent,
    env_kwargs=ddpg_env_kwargs,
    best_seed=ddpg_best_seed,
)

"""# Part (f): Exploration and Action Noise Study (DDPG-Specific)

## OU Noise: ($\theta=0.2, \sigma=0.15$)
"""

cfg = DDPGConfig(
    gamma=0.95,
    tau=0.01,
    actor_lr=3e-4,
    critic_lr=1e-3,
    batch_size=64,
    buffer_capacity=100_000,
    warmup_steps=2000,
    train_every=1,
    gradient_steps=2,
    ou_theta=0.20,
    ou_sigma=0.15,
    noise_type="ou",
)

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode=circular_target_fn,
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ddpg_agent = DDPGAgent(state_dim, action_dim, 2.0, cfg, device="cuda")
# train
ddpg_logs = ddpg_agent.train(env, n_episodes=100, seed=7, verbose_every=25)

plot_logs_grid(ddpg_logs, name_map={
    "ret": "Episode Return",
    "mean_tracking_error": "Mean Tracking Error",
    "success_rate": "Success Rate",
    "control_energy": "Control Energy",
    "smoothness": "Torque Smoothness",
}, title="DDPG Training Metrics")

# evaluate
ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=2025)

print("\nFinal Evaluation Metrics")
for k, v in ddpg_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

""" ## OU Noise: ($\theta=0.2, \sigma=0.25$)"""

cfg = DDPGConfig(
    gamma=0.95,
    tau=0.01,
    actor_lr=3e-4,
    critic_lr=1e-3,
    batch_size=64,
    buffer_capacity=100_000,
    warmup_steps=2000,
    train_every=1,
    gradient_steps=2,
    ou_theta=0.20,
    ou_sigma=0.25,
    noise_type="ou",
)

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode=circular_target_fn,
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ddpg_agent = DDPGAgent(state_dim, action_dim, 2.0, cfg, device="cuda")
# train
ddpg_logs = ddpg_agent.train(env, n_episodes=100, seed=7, verbose_every=25)

plot_logs_grid(ddpg_logs, name_map={
    "ret": "Episode Return",
    "mean_tracking_error": "Mean Tracking Error",
    "success_rate": "Success Rate",
    "control_energy": "Control Energy",
    "smoothness": "Torque Smoothness",
}, title="DDPG Training Metrics")

# evaluate
ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=2025)

print("\nFinal Evaluation Metrics")
for k, v in ddpg_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

""" ## OU Noise: ($\theta=0.1, \sigma=0.25$)"""

cfg = DDPGConfig(
    gamma=0.95,
    tau=0.01,
    actor_lr=3e-4,
    critic_lr=1e-3,
    batch_size=64,
    buffer_capacity=100_000,
    warmup_steps=2000,
    train_every=1,
    gradient_steps=2,
    ou_theta=0.10,
    ou_sigma=0.25,
    noise_type="ou",
)

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode=circular_target_fn,
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ddpg_agent = DDPGAgent(state_dim, action_dim, 2.0, cfg, device="cuda")
# train
ddpg_logs = ddpg_agent.train(env, n_episodes=100, seed=7, verbose_every=25)

plot_logs_grid(ddpg_logs, name_map={
    "ret": "Episode Return",
    "mean_tracking_error": "Mean Tracking Error",
    "success_rate": "Success Rate",
    "control_energy": "Control Energy",
    "smoothness": "Torque Smoothness",
}, title="DDPG Training Metrics")

# evaluate
ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=2025)

print("\nFinal Evaluation Metrics")
for k, v in ddpg_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""## Gaussian Noise: ($\sigma=0.15$)"""

cfg = DDPGConfig(
    gamma=0.95,
    tau=0.01,
    actor_lr=3e-4,
    critic_lr=1e-3,
    batch_size=64,
    buffer_capacity=100_000,
    warmup_steps=2000,
    train_every=1,
    gradient_steps=2,
    ou_theta=0.10,
    ou_sigma=0.15,
    noise_type="gaussian",
)

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode=circular_target_fn,
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ddpg_agent = DDPGAgent(state_dim, action_dim, 2.0, cfg, device="cuda")
# train
ddpg_logs = ddpg_agent.train(env, n_episodes=100, seed=7, verbose_every=25)

plot_logs_grid(ddpg_logs, name_map={
    "ret": "Episode Return",
    "mean_tracking_error": "Mean Tracking Error",
    "success_rate": "Success Rate",
    "control_energy": "Control Energy",
    "smoothness": "Torque Smoothness",
}, title="DDPG Training Metrics")

# evaluate
ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=2025)

print("\nFinal Evaluation Metrics")
for k, v in ddpg_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""## Gaussian Noise: ($\sigma=0.45$)"""

cfg = DDPGConfig(
    gamma=0.95,
    tau=0.01,
    actor_lr=3e-4,
    critic_lr=1e-3,
    batch_size=64,
    buffer_capacity=100_000,
    warmup_steps=2000,
    train_every=1,
    gradient_steps=2,
    ou_theta=0.10,
    ou_sigma=0.45,
    noise_type="gaussian",
)

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode=circular_target_fn,
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ddpg_agent = DDPGAgent(state_dim, action_dim, 2.0, cfg, device="cuda")
# train
ddpg_logs = ddpg_agent.train(env, n_episodes=100, seed=7, verbose_every=25)

plot_logs_grid(ddpg_logs, name_map={
    "ret": "Episode Return",
    "mean_tracking_error": "Mean Tracking Error",
    "success_rate": "Success Rate",
    "control_energy": "Control Energy",
    "smoothness": "Torque Smoothness",
}, title="DDPG Training Metrics")

# evaluate
ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=2025)

print("\nFinal Evaluation Metrics")
for k, v in ddpg_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""# Part (g): Generalization Tests (Target Distribution Shift)

## 10-Step SARSA Trained on Circular Moving Target with $w=\frac{4}{30}\pi, r=0.5 , (x_c.y_c)=(0,0)$
"""

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

# ============================================================
# Train
# ============================================================

sarsa_logs = sarsa_agent.train(
    env=env,
    n_step=10,
    gamma=0.95,
    alpha=0.17,
    n_episodes=100,
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

"""### Evaluated on Circular Moving Target with $w=\frac{4}{30}\pi, r=0.5 , (x_c.y_c)=(0,0)$"""

def my_target1(t):
    x_c = 0.0
    y_c = 0.0
    r = 0.5
    w = 4 / 30 * np.pi
    return np.array([
        x_c + r * np.cos(w * t),
        y_c + r * np.sin(w * t)
    ], dtype=np.float64)

base_env = TwoLinkReacherEnv(
    dt=0.02,
    obs_mode="minimal",
    target_mode=my_target1,
    max_steps=150,
    torque_limit=2.0,
    reward_torque_penalty=0.0,
    render_mode=None,
    seed=42,
)

env = DiscreteTorqueWrapper(base_env)

sarsa_metrics, sarsa_best_seed = sarsa_agent.evaluate(
    env=env,
    n_episodes=100,
    epsilon_eval=0.0,
    seed=2025,
)

print("\nFinal Evaluation Metrics")
for k, v in sarsa_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""### Evaluated on Circular Moving Target with $w=\frac{1}{3}\pi, r=0.5 , (x_c.y_c)=(0,0)$"""

def my_target2(t):
    x_c = 0.0
    y_c = 0.0
    r = 0.5
    w = 1 / 3 * np.pi
    return np.array([
        x_c + r * np.cos(w * t),
        y_c + r * np.sin(w * t)
    ], dtype=np.float64)

base_env = TwoLinkReacherEnv(
    dt=0.02,
    obs_mode="minimal",
    target_mode=my_target2,
    max_steps=150,
    torque_limit=2.0,
    reward_torque_penalty=0.0,
    render_mode=None,
    seed=42,
)

env = DiscreteTorqueWrapper(base_env)

sarsa_metrics, sarsa_best_seed = sarsa_agent.evaluate(
    env=env,
    n_episodes=100,
    epsilon_eval=0.0,
    seed=2025,
)

print("\nFinal Evaluation Metrics")
for k, v in sarsa_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""### Evaluated on Circular Moving Target with $w=\frac{4}{30}\pi, r=1.5 , (x_c.y_c)=(0,0)$"""

def my_target3(t):
    x_c = 0.0
    y_c = 0.0
    r = 1.5
    w = 4 / 30 * np.pi
    return np.array([
        x_c + r * np.cos(w * t),
        y_c + r * np.sin(w * t)
    ], dtype=np.float64)

base_env = TwoLinkReacherEnv(
    dt=0.02,
    obs_mode="minimal",
    target_mode=my_target3,
    max_steps=150,
    torque_limit=2.0,
    reward_torque_penalty=0.0,
    render_mode=None,
    seed=42,
)

env = DiscreteTorqueWrapper(base_env)

sarsa_metrics, sarsa_best_seed = sarsa_agent.evaluate(
    env=env,
    n_episodes=100,
    epsilon_eval=0.0,
    seed=2025,
)

print("\nFinal Evaluation Metrics")
for k, v in sarsa_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""### Evaluated on Lissajous"""

base_env = TwoLinkReacherEnv(
    dt=0.02,
    obs_mode="minimal",
    target_mode="lissajous",
    max_steps=150,
    torque_limit=2.0,
    reward_torque_penalty=0.0,
    render_mode=None,
    seed=42,
)

env = DiscreteTorqueWrapper(base_env)

sarsa_metrics, sarsa_best_seed = sarsa_agent.evaluate(
    env=env,
    n_episodes=100,
    epsilon_eval=0.0,
    seed=2025,
)

print("\nFinal Evaluation Metrics")
for k, v in sarsa_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""## DDPG Trained on Circular Moving Target with $w=\frac{4}{30}\pi, r=0.5 , (x_c.y_c)=(0,0)$"""

cfg = DDPGConfig(
    gamma=0.95,
    tau=0.01,
    actor_lr=3e-4,
    critic_lr=1e-3,
    batch_size=64,
    buffer_capacity=100_000,
    warmup_steps=2000,
    train_every=1,
    gradient_steps=2,
    ou_theta=0.10,
    ou_sigma=0.15,
)

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode=circular_target_fn,
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ddpg_agent = DDPGAgent(state_dim, action_dim, 2.0, cfg, device="cuda")
# train
ddpg_logs = ddpg_agent.train(env, n_episodes=100, seed=7, verbose_every=25)

plot_logs_grid(ddpg_logs, name_map={
    "ret": "Episode Return",
    "mean_tracking_error": "Mean Tracking Error",
    "success_rate": "Success Rate",
    "control_energy": "Control Energy",
    "smoothness": "Torque Smoothness",
}, title="DDPG Training Metrics")

"""### Evaluated on Circular Moving Target with $w=\frac{4}{30}\pi, r=0.5 , (x_c.y_c)=(0,0)$"""

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode=my_target1,
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

# evaluate
ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=2025)

print("\nFinal Evaluation Metrics")
for k, v in ddpg_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""### Evaluated on Circular Moving Target with $w=\frac{1}{3}\pi, r=0.5 , (x_c.y_c)=(0,0)$"""

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode=my_target2,
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

# evaluate
ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=2025)

print("\nFinal Evaluation Metrics")
for k, v in ddpg_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""### Evaluated on Circular Moving Target with $w=\frac{4}{30}\pi, r=1.5 , (x_c.y_c)=(0,0)$"""

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode=my_target3,
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

# evaluate
ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=2025)

print("\nFinal Evaluation Metrics")
for k, v in ddpg_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""### Evaluated on Lissajous"""

env = TwoLinkReacherEnv(
    dt=0.02,
    render_mode=None,
    obs_mode="minimal_ee",        # REQUIRED format
    max_steps=150,
    torque_limit=2.0,
    target_mode="lissajous",
    reward_torque_penalty=0.0,
    success_radius=None,
    draw_trail=False,
    seed=42,
)

# evaluate
ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=2025)

print("\nFinal Evaluation Metrics")
for k, v in ddpg_metrics.items():
    print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

"""| Method        | Train target                     | Eval target             |  Avg return | Std return | Avg mean tracking error | Avg success rate | Avg control energy | Avg smoothness | Avg final distance |
| ------------- | -------------------------------- | ----------------------- | ----------: | ---------: | ----------------------: | ---------------: | -----------------: | -------------: | -----------------: |
| 10-step SARSA | Circle (ω=4π/30, r=0.5, c=(0,0)) | Circle (ω=4π/30, r=0.5) | -331.734752 |   1.478107 |                1.333531 |         0.000000 |         924.200000 |     609.160000 |           1.547449 |
| 10-step SARSA | Circle (ω=4π/30, r=0.5, c=(0,0)) | Circle (ω=π/3, r=0.5)   | -144.661778 |   4.640882 |                0.915761 |         0.000000 |         827.120000 |     924.080000 |           1.435190 |
| 10-step SARSA | Circle (ω=4π/30, r=0.5, c=(0,0)) | Circle (ω=4π/30, r=1.5) | -241.796888 |  21.784259 |                1.178509 |         0.000000 |         615.000000 |    1215.200000 |           1.924927 |
| 10-step SARSA | Circle (ω=4π/30, r=0.5, c=(0,0)) | Lissajous               | -547.826250 |  93.623974 |                1.797516 |         0.001533 |         830.960000 |    1311.480000 |           1.009500 |
| DDPG          | Circle (ω=4π/30, r=0.5, c=(0,0)) | Circle (ω=4π/30, r=0.5) |  -64.777439 |   0.000000 |                0.457595 |         0.300000 |         542.509129 |       6.505330 |           0.115069 |
| DDPG          | Circle (ω=4π/30, r=0.5, c=(0,0)) | Circle (ω=π/3, r=0.5)   |  -94.900051 |   0.000000 |                0.712246 |         0.000000 |         829.602720 |      22.342715 |           0.744311 |
| DDPG          | Circle (ω=4π/30, r=0.5, c=(0,0)) | Circle (ω=4π/30, r=1.5) | -129.279111 |   0.000000 |                0.841923 |         0.000000 |         676.299277 |      10.052140 |           0.608443 |
| DDPG          | Circle (ω=4π/30, r=0.5, c=(0,0)) | Lissajous               | -188.447931 |   0.000000 |                1.076471 |         0.026667 |         578.322415 |       6.255174 |           1.008789 |

In my experiments, I trained both a 10-step SARSA agent (with discretized torques and linear tile-coded value approximation) and a DDPG agent (continuous torques with neural actor–critic) on the same circular moving target defined by ω = 4π/30, r = 0.5, centered at the origin. I then evaluated each learned policy not only on the training circle but also under distribution shifts that changed the angular speed (ω = π/3), changed the radius (r = 1.5), and changed the trajectory family entirely (a Lissajous curve). The table summarizes the evaluation metrics across 100 episodes for each condition, including return, mean tracking error, success rate, control energy, smoothness, and final distance.

On the in-distribution evaluation (same circle as training), DDPG clearly learned an effective tracking policy while SARSA did not. DDPG achieved an average return of -64.78 with a mean tracking error of 0.458 and a 0.30 success rate, while also maintaining low control energy (542.5) and very low torque variation (smoothness 6.51). In contrast, SARSA’s performance remained poor on the training circle, with an average return of -331.73, a mean tracking error of 1.334, and a zero success rate. The energy and smoothness numbers for SARSA were also much higher (924.2 energy, 609.2 smoothness), which is consistent with a coarse, switching torque policy that fails to stabilize close tracking around a moving target.

When I tested generalization by increasing the target speed to ω = π/3, both agents degraded, but the degradation looked qualitatively different. DDPG’s mean tracking error rose from 0.458 to 0.712 and success collapsed from 0.30 to 0.00, while energy and smoothness increased sharply (829.6 energy and 22.34 smoothness), indicating that the learned policy had to apply more aggressive and less smooth torques to keep up with the faster-moving target, yet still could not remain within the success threshold. SARSA, meanwhile, reported a less negative return (-144.66) and a lower mean tracking error (0.916) than on the training circle, but success remained 0.00 and smoothness worsened substantially (924.1). I interpret this as a failure-mode artifact rather than true transfer: because SARSA never learned a competent tracker on the training distribution, apparent “improvements” under a different ω can happen simply because the target motion happens to align better with the agent’s crude limit-cycle behavior, not because the policy has learned invariances.

Changing the radius to r = 1.5 similarly exposed specialization in the learned policies. DDPG’s tracking error increased further to 0.842 with success still at 0.00, and its energy and smoothness moved to intermediate values (676.3 energy, 10.05 smoothness). This suggests that larger-radius motion likely pushed the arm into regions requiring different torque patterns and timing, and the policy trained on r = 0.5 was not robust to that shift, although it remained relatively smooth compared to SARSA. SARSA again failed to achieve any success, and its smoothness became extremely large (1215.2), consistent with frequent switching among the discrete torque actions without converging to stable tracking.

The trajectory-family shift to Lissajous provides the clearest view of “skill transfer” versus specialization. DDPG retained some partial competence, with a non-zero success rate of 0.0267, mean tracking error of 1.076, and low smoothness (6.26), implying it learned a generally stabilizing continuous feedback behavior that sometimes brings the end-effector close to the moving target even under a qualitatively different path. SARSA’s performance on Lissajous remained weak: the mean tracking error rose to 1.798 and success was only 0.00153, while energy and smoothness were again very large (831.0 and 1311.5). Overall, these results indicate that DDPG learns a high-quality in-distribution controller and exhibits limited but real out-of-distribution generalization, whereas the SARSA setup, constrained by a coarse discrete action set and a linear approximator, does not learn reliable tracking and therefore cannot demonstrate meaningful generalization under target dynamics shifts.

# Part (h): Ablation Study on State Inputs

## DDPG Trained on Circular Moving Target with $w=\frac{4}{30}\pi, r=0.5 , (x_c.y_c)=(0,0)$, with Minimal State Definition
"""

def ablation_test(seed, obs_mode):
    print(100 * "*")
    print(f"Seed={seed}, State Definition={obs_mode}!\n")
    cfg = DDPGConfig(
        gamma=0.95,
        tau=0.01,
        actor_lr=3e-4,
        critic_lr=1e-3,
        batch_size=64,
        buffer_capacity=100_000,
        warmup_steps=2000,
        train_every=1,
        gradient_steps=2,
        ou_theta=0.10,
        ou_sigma=0.15,
    )

    env = TwoLinkReacherEnv(
        dt=0.02,
        render_mode=None,
        obs_mode=obs_mode,        # REQUIRED format
        max_steps=150,
        torque_limit=2.0,
        target_mode="static_random",
        reward_torque_penalty=0.0,
        success_radius=None,
        draw_trail=False,
        seed=seed,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ddpg_agent = DDPGAgent(state_dim, action_dim, 2.0, cfg, device="cuda")
    # train
    ddpg_logs = ddpg_agent.train(env, n_episodes=100, seed=seed, verbose_every=25)

    plot_logs_grid(ddpg_logs, name_map={
        "ret": "Episode Return",
        "mean_tracking_error": "Mean Tracking Error",
        "success_rate": "Success Rate",
        "control_energy": "Control Energy",
        "smoothness": "Torque Smoothness",
    }, title="DDPG Training Metrics")

    # evaluate
    ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=seed+1000)

    print("\nFinal Evaluation Metrics")
    for k, v in ddpg_metrics.items():
        print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

for k in range(3):
    ablation_test(100 + k, "minimal")

"""## DDPG Trained on Circular Moving Target with $w=\frac{4}{30}\pi, r=0.5 , (x_c.y_c)=(0,0)$, with Minimal+end-effector State Definition"""

for k in range(3):
    ablation_test(100 + k, "minimal_ee")

"""## DDPG Trained on Circular Moving Target with $w=\frac{4}{30}\pi, r=0.5 , (x_c.y_c)=(0,0)$, with Error-only State Definition"""

for k in range(3):
    ablation_test(100 + k, "error_only")

"""## DDPG Trained on Circular Moving Target with $w=\frac{4}{30}\pi, r=0.5 , (x_c.y_c)=(0,0)$, with Error in Inverse-Kinematics State Definition"""

for k in range(3):
    ablation_test(100 + k, "ik_error")

"""Final table (DDPG trained on circular target; varying state/observation representation; eval metrics)

| State definition | Seed |  Avg return | Std return | Avg mean tracking error | Avg success rate | Avg control energy | Avg smoothness |
| ---------------- | ---: | ----------: | ---------: | ----------------------: | ---------------: | -----------------: | -------------: |
| minimal          |  100 | -436.349618 | 241.616015 |                1.527686 |         0.003800 |         657.391382 |      14.103744 |
| minimal          |  101 | -493.380176 | 286.098282 |                1.626748 |         0.003200 |         574.974667 |      17.874367 |
| minimal          |  102 | -464.237997 | 318.682364 |                1.567357 |         0.003467 |         644.366707 |      13.733257 |
| minimal_ee       |  100 | -397.590820 | 225.010881 |                1.440078 |         0.008000 |         666.134031 |      14.195062 |
| minimal_ee       |  101 | -362.244375 | 200.201695 |                1.368055 |         0.005933 |         557.169911 |      15.693691 |
| minimal_ee       |  102 | -327.640564 | 206.197895 |                1.275923 |         0.012267 |         669.127124 |      13.707850 |
| error_only       |  100 | -498.582683 | 257.551557 |                1.616634 |         0.005200 |         756.222239 |      26.112926 |
| error_only       |  101 | -290.820254 | 174.024081 |                1.195013 |         0.013267 |         561.142137 |      12.485145 |
| error_only       |  102 | -409.140083 | 245.391990 |                1.475314 |         0.002933 |         681.234569 |      13.832367 |
| ik_error         |  100 | -441.323775 | 294.861661 |                1.502656 |         0.026667 |         518.807631 |      16.635480 |
| ik_error         |  101 | -523.398766 | 345.969015 |                1.677261 |         0.000733 |         541.165557 |      19.000687 |
| ik_error         |  102 | -515.796541 | 401.559791 |                1.599451 |         0.011533 |         542.556804 |      94.332270 |

Seed-averaged summary by representation (mean across seeds 100/101/102)

| State definition | Mean avg return | Mean std return | Mean tracking error | Mean success rate | Mean control energy | Mean smoothness |
| ---------------- | --------------: | --------------: | ------------------: | ----------------: | ------------------: | --------------: |
| minimal          |        -464.656 |         282.132 |               1.574 |           0.00349 |             625.578 |          15.237 |
| minimal_ee       |        -362.492 |         210.470 |               1.361 |           0.00873 |             630.810 |          14.532 |
| error_only       |        -399.514 |         225.656 |               1.429 |           0.00713 |             666.200 |          17.477 |
| ik_error         |        -493.506 |         347.463 |               1.593 |           0.01298 |             534.177 |          43.323 |

When I compare state/observation representations for DDPG trained and evaluated on the circular moving target, the clearest result is that the representation strongly affects both performance and run-to-run stability. Looking at the seed-averaged summary, `minimal_ee` is the most consistently effective choice: it achieves the best mean return (-362.492), the lowest mean tracking error (1.361), and the lowest mean variability in return (std_return 210.470). In other words, across seeds 100–102 it not only performs better on average, but it also behaves more predictably than the other observation definitions.

The baseline `minimal` representation is consistently weaker. Its mean return is much worse (-464.656), tracking error is higher (1.574), and the average success rate is the lowest of all four options (0.00349). That pattern is also reflected at the per-seed level: all three `minimal` runs cluster around very negative returns (roughly -436 to -493) and relatively large tracking errors (about 1.53–1.63). This suggests that, with `minimal`, the policy is effectively being asked to infer more of the task-relevant geometry internally, which makes learning harder and yields weaker tracking under the same training budget.

`error_only` sits in the middle in terms of averages, but it is noticeably less reliable. Its mean return (-399.514) and mean tracking error (1.429) are better than `minimal` but still worse than `minimal_ee`, and its mean success rate (0.00713) is close to `minimal_ee` (0.00873). The per-seed results show why I treat it as less dependable: seed 101 is extremely strong relative to the rest of the table (return -290.820, tracking error 1.195, success 0.013267, low smoothness 12.485), but seeds 100 and 102 are substantially worse. So `error_only` can produce the best single run, but it does not consistently do so across seeds.

Finally, `ik_error` is the most unstable representation in this sweep. It has the worst mean return (-493.506) and one of the highest mean tracking errors (1.593), even though it posts the highest mean success rate (0.01298). That apparent advantage in success rate is not robust: it is driven mainly by seed 100 (success 0.026667), while seed 101 collapses (0.000733), and seed 102 shows extreme torque oscillation (smoothness 94.332), which dominates the mean smoothness (43.323). Overall, based on both the seed-level table and the seed-averaged summary, `minimal_ee` is the best representation to report as the strongest and most reliable choice, `error_only` is a high-variance runner-up, and `minimal` and `ik_error` are clearly inferior for this setup.

# Part (i): Metrics and Reporting Standards

## 10-Step SARSA Training with $K=3$ Different Seeds

## Static Fixed Target
"""

def train_and_eval_sarsa_with_seeds(seed):
    print(100 * "*")
    print(f"Seed={seed}!\n")
    base_env = TwoLinkReacherEnv(
        dt=0.02,
        obs_mode="minimal",
        target_mode=static_target_fn,
        max_steps=150,
        torque_limit=2.0,
        reward_torque_penalty=0.0,
        render_mode=None,
        seed=seed,
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

    # ============================================================
    # Train
    # ============================================================

    sarsa_logs = sarsa_agent.train(
        env=env,
        n_step=10,
        gamma=0.95,
        alpha=0.17,
        n_episodes=100,
        eps_start=0.995,
        eps_end=0.03,
        eps_decay_frac=0.65,
        seed=seed,
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
        seed=seed+1000,
    )

    print("\nFinal Evaluation Metrics")
    for k, v in sarsa_metrics.items():
        print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

for k in range(3):
    train_and_eval_sarsa_with_seeds(k+10000)

"""## Circular Moving Target"""

def train_and_eval_sarsa_with_seeds(seed):
    print(100 * "*")
    print(f"Seed={seed}!\n")
    base_env = TwoLinkReacherEnv(
        dt=0.02,
        obs_mode="minimal",
        target_mode=circular_target_fn,
        max_steps=150,
        torque_limit=2.0,
        reward_torque_penalty=0.0,
        render_mode=None,
        seed=seed,
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

    # ============================================================
    # Train
    # ============================================================

    sarsa_logs = sarsa_agent.train(
        env=env,
        n_step=10,
        gamma=0.95,
        alpha=0.17,
        n_episodes=100,
        eps_start=0.995,
        eps_end=0.03,
        eps_decay_frac=0.65,
        seed=seed,
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
        seed=seed+1000,
    )

    print("\nFinal Evaluation Metrics")
    for k, v in sarsa_metrics.items():
        print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

for k in range(3):
    train_and_eval_sarsa_with_seeds(k+10000)

"""## DDPG Training with $K=3$ Different Seeds

## Static Fixed Target
"""

def train_and_eval_ddpg_with_seeds(seed):
    print(100 * "*")
    print(f"Seed={seed}!\n")
    cfg = DDPGConfig(
        gamma=0.95,
        tau=0.01,
        actor_lr=3e-4,
        critic_lr=1e-3,
        batch_size=64,
        buffer_capacity=100_000,
        warmup_steps=2000,
        train_every=1,
        gradient_steps=2,
        ou_theta=0.10,
        ou_sigma=0.15,
    )

    env = TwoLinkReacherEnv(
        dt=0.02,
        render_mode=None,
        obs_mode="minimal_ee",        # REQUIRED format
        max_steps=150,
        torque_limit=2.0,
        target_mode=static_target_fn,
        reward_torque_penalty=0.0,
        success_radius=None,
        draw_trail=False,
        seed=seed,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ddpg_agent = DDPGAgent(state_dim, action_dim, 2.0, cfg, device="cuda")
    # train
    ddpg_logs = ddpg_agent.train(env, n_episodes=100, seed=seed, verbose_every=25)

    plot_logs_grid(ddpg_logs, name_map={
        "ret": "Episode Return",
        "mean_tracking_error": "Mean Tracking Error",
        "success_rate": "Success Rate",
        "control_energy": "Control Energy",
        "smoothness": "Torque Smoothness",
    }, title="DDPG Training Metrics")

    # evaluate
    ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=seed+1000)

    print("\nFinal Evaluation Metrics")
    for k, v in ddpg_metrics.items():
        print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

for k in range(3):
    train_and_eval_ddpg_with_seeds(k+10000)

"""## Circular Moving Target"""

def train_and_eval_ddpg_with_seeds(seed):
    print(100 * "*")
    print(f"Seed={seed}!\n")
    cfg = DDPGConfig(
        gamma=0.95,
        tau=0.01,
        actor_lr=3e-4,
        critic_lr=1e-3,
        batch_size=64,
        buffer_capacity=100_000,
        warmup_steps=2000,
        train_every=1,
        gradient_steps=2,
        ou_theta=0.10,
        ou_sigma=0.15,
    )

    env = TwoLinkReacherEnv(
        dt=0.02,
        render_mode=None,
        obs_mode="minimal_ee",        # REQUIRED format
        max_steps=150,
        torque_limit=2.0,
        target_mode=circular_target_fn,
        reward_torque_penalty=0.0,
        success_radius=None,
        draw_trail=False,
        seed=seed,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ddpg_agent = DDPGAgent(state_dim, action_dim, 2.0, cfg, device="cuda")
    # train
    ddpg_logs = ddpg_agent.train(env, n_episodes=100, seed=seed, verbose_every=25)

    plot_logs_grid(ddpg_logs, name_map={
        "ret": "Episode Return",
        "mean_tracking_error": "Mean Tracking Error",
        "success_rate": "Success Rate",
        "control_energy": "Control Energy",
        "smoothness": "Torque Smoothness",
    }, title="DDPG Training Metrics")

    # evaluate
    ddpg_metrics, ddpg_best_seed = ddpg_agent.evaluate(env, n_episodes=100, seed=seed+1000)

    print("\nFinal Evaluation Metrics")
    for k, v in ddpg_metrics.items():
        print(f"{k:>20s}: {v:.6f}" if isinstance(v, float) else f"{k:>20s}: {v}")

for k in range(3):
    train_and_eval_ddpg_with_seeds(k+10000)

"""| Method                                      | Task / target                        | Avg return (mean ± SD) | Mean tracking error (mean ± SD) | Success rate (mean ± SD) | Control energy (mean ± SD) | Smoothness (mean ± SD) |
| ------------------------------------------- | ------------------------------------ | ---------------------: | ------------------------------: | -----------------------: | -------------------------: | ---------------------: |
| PID (Cartesian PID + Jᵀ)                    | Static target (1.5, 0)               |                    N/A |                        0.221255 |                 0.566667 |                 197.173454 |               1.607569 |
| PID (Cartesian PID + Jᵀ)                    | Circle (r=0.5, c=(0,0))              |                    N/A |                        0.532498 |                 0.393333 |                 238.393026 |               3.833921 |
| Joint PID + IK                              | Static target (1.5, 0)               |                    N/A |                        0.133525 |                 0.280000 |                 149.979292 |               2.045299 |
| Joint PID + IK                              | Circle (r=0.5, c=(0,0))              |                    N/A |                        0.407195 |                 0.266667 |                 500.528523 |               2.521114 |
| n-step SARSA (tile-coded, discrete torques) | Static target (3 seeds: 10000–10002) |      -117.944 ± 83.833 |                   0.712 ± 0.336 |            0.064 ± 0.112 |           736.267 ± 60.770 |       553.227 ± 99.029 |
| n-step SARSA (tile-coded, discrete torques) | Circle (3 seeds: 10000–10002)        |     -241.278 ± 106.234 |                   1.130 ± 0.272 |            0.013 ± 0.023 |           876.000 ± 10.583 |      730.667 ± 213.629 |
| DDPG (continuous torques)                   | Static target (3 seeds: 10000–10002) |      -69.230 ± 105.360 |                   0.441 ± 0.425 |            0.267 ± 0.249 |          400.782 ± 239.914 |          6.798 ± 3.436 |
| DDPG (continuous torques)                   | Circle (3 seeds: 10000–10002)        |     -152.160 ± 122.378 |                   0.854 ± 0.380 |            0.036 ± 0.037 |          764.840 ± 234.623 |        29.079 ± 28.778 |

Notes for interpreting the table: PID and PID+IK metrics are single runs (no seed sweep provided), while SARSA and DDPG rows are the mean and standard deviation across the three seeds reported.
"""