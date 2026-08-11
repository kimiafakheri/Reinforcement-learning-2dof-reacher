
# ============================================================
# Cartesian PID + Jacobian Transpose Controller
# ============================================================

class CartesianPIDController:
    """
    PID in Cartesian space:
        e = p_target - p_end
        u = Kp e + Ki ∫e dt + Kd de/dt
        tau = J(q)^T u
    """

    def __init__(
        self,
        l1=1.0,
        l2=1.0,
        kp=(20.0, 20.0),
        ki=(0.5, 0.5),
        kd=(6.0, 6.0),
        dt=0.02,
        torque_limit=2.0,
        integral_clip=2.0,
    ):
        self.l1 = l1
        self.l2 = l2
        self.Kp = np.diag(np.asarray(kp, dtype=np.float64))
        self.Ki = np.diag(np.asarray(ki, dtype=np.float64))
        self.Kd = np.diag(np.asarray(kd, dtype=np.float64))
        self.dt = float(dt)
        self.torque_limit = float(torque_limit)
        self.integral_clip = float(integral_clip)

        self.reset()

    def reset(self):
        self.e_int = np.zeros(2, dtype=np.float64)
        self.e_prev = np.zeros(2, dtype=np.float64)
        self.first = True

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        th1, th2 = q
        x = self.l1 * np.cos(th1) + self.l2 * np.cos(th1 + th2)
        y = self.l1 * np.sin(th1) + self.l2 * np.sin(th1 + th2)
        return np.array([x, y], dtype=np.float64)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        th1, th2 = q
        s1 = np.sin(th1)
        c1 = np.cos(th1)
        s12 = np.sin(th1 + th2)
        c12 = np.cos(th1 + th2)

        J = np.array([
            [-self.l1 * s1 - self.l2 * s12, -self.l2 * s12],
            [ self.l1 * c1 + self.l2 * c12,  self.l2 * c12],
        ], dtype=np.float64)
        return J

    def act(self, q: np.ndarray, dq: np.ndarray, p_target: np.ndarray) -> tuple[np.ndarray, dict]:
        p_end = self.forward_kinematics(q)
        e = p_target - p_end

        # Integral term
        self.e_int += e * self.dt
        self.e_int = np.clip(self.e_int, -self.integral_clip, self.integral_clip)

        # Derivative term
        if self.first:
            de = np.zeros_like(e)
            self.first = False
        else:
            de = (e - self.e_prev) / self.dt
        self.e_prev = e.copy()

        # Cartesian "force-like" control
        u = self.Kp @ e + self.Ki @ self.e_int + self.Kd @ de

        # Jacobian transpose torque mapping
        J = self.jacobian(q)
        tau = J.T @ u

        # Clip to actuator limits
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)

        info = {
            "p_end": p_end,
            "p_target": p_target,
            "e_vec": e,
            "e_norm": float(np.linalg.norm(e)),
            "u": u,
            "J": J,
        }
        return tau, info


# ============================================================
# Evaluation metrics
# ============================================================

def compute_metrics(errors: list[float], taus: list[np.ndarray], epsilon: float = 0.05) -> dict:
    """
    Required metrics:
      1) Mean tracking error
      2) Success rate
      3) Control energy
      4) Smoothness
    """
    e_arr = np.asarray(errors, dtype=np.float64)                  # shape (T,)
    tau_arr = np.asarray(taus, dtype=np.float64)                  # shape (T,2)

    mean_tracking_error = float(np.mean(e_arr))

    success_rate = float(np.mean(e_arr < epsilon))

    control_energy = float(np.sum(np.sum(tau_arr ** 2, axis=1)))

    if len(tau_arr) >= 2:
        dtau = tau_arr[1:] - tau_arr[:-1]
        smoothness = float(np.sum(np.sum(dtau ** 2, axis=1)))
    else:
        smoothness = 0.0

    return {
        "mean_tracking_error": mean_tracking_error,
        "success_rate": success_rate,
        "control_energy": control_energy,
        "smoothness": smoothness,
    }


# ============================================================
# Rollout / experiment runner
# ============================================================

def run_pid_episode(
    target_mode,
    video_folder: str | None = None,
    video_name: str = "pid_rollout",
    render_video: bool = False,
    seed: int = 0,
    max_steps: int = 150,
    dt: float = 0.02,
    torque_limit: float = 2.0,
    epsilon_success: float = 0.05,
    gains=None,
):
    """
    Runs one rollout and returns metrics + traces.
    """
    if gains is None:
        gains = {
            "kp": (20.0, 20.0),
            "ki": (0.5, 0.5),
            "kd": (6.0, 6.0),
        }

    render_mode = "rgb_array" if render_video else None

    env = TwoLinkReacherEnv(
        dt=dt,
        render_mode=render_mode,
        obs_mode="minimal",
        max_steps=max_steps,
        torque_limit=torque_limit,
        target_mode=target_mode,
        success_radius=None,
        reward_torque_penalty=0.0,
        draw_trail=True,
        trail_len=80,
        random_init=True
    )

    if render_video and video_folder is not None:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep: True,
            name_prefix=video_name,
        )

    ctrl = CartesianPIDController(
        l1=1.0,
        l2=1.0,
        kp=gains["kp"],
        ki=gains["ki"],
        kd=gains["kd"],
        dt=dt,
        torque_limit=torque_limit,
        integral_clip=2.0,
    )

    obs, info = env.reset(seed=seed)
    ctrl.reset()

    errors = []
    taus = []
    p_ends = []
    p_targets = []

    terminated = truncated = False
    while not (terminated or truncated):
        base_env = env.unwrapped

        q = base_env.q.copy()
        dq = base_env.dq.copy()
        p_target = base_env.target.copy()

        tau, cinfo = ctrl.act(q=q, dq=dq, p_target=p_target)

        obs, reward, terminated, truncated, info = env.step(tau)

        errors.append(cinfo["e_norm"])
        taus.append(tau.copy())
        p_ends.append(cinfo["p_end"].copy())
        p_targets.append(cinfo["p_target"].copy())

    env.close()

    metrics = compute_metrics(errors, taus, epsilon=epsilon_success)

    traces = {
        "errors": np.asarray(errors),
        "taus": np.asarray(taus),
        "p_end": np.asarray(p_ends),
        "p_target": np.asarray(p_targets),
    }

    return metrics, traces

gains = {
    "kp": (14.0, 14.0),
    "ki": (0.2, 0.2),
    "kd": (10.0, 10.0),
}

# 1) Static target scenario
static_metrics, _ = run_pid_episode(
    target_mode=static_target_fn,
    render_video=True,
    video_folder="videos_pid",
    video_name="pid_static",
    seed=42,
    gains=gains,
    epsilon_success=0.15,
)

# 2) Circular target scenario
circ_metrics, _ = run_pid_episode(
    target_mode=circular_target_fn,
    render_video=True,
    video_folder="videos_pid",
    video_name="pid_circle",
    seed=42,
    gains=gains,
    epsilon_success=0.15,
)

print("\n=== PID Cartesian + Jacobian-Transpose Results ===")
print("Gains:", gains)

print("\nStatic target (1.5, 0):")
for k, v in static_metrics.items():
    print(f"  {k}: {v:.6f}")

video_dir = "videos_pid"
video_path = os.path.join(video_dir, "pid_static-episode-0.mp4")
print(f"\nDisplaying: {video_path}")
display(Video(video_path, embed=True))

print("\nCircular target (r=0.5, center=(0,0)):")
for k, v in circ_metrics.items():
    print(f"  {k}: {v:.6f}")

video_path = os.path.join(video_dir, "pid_circle-episode-0.mp4")
print(f"\nDisplaying: {video_path}")
display(Video(video_path, embed=True))

print("\nVideos saved in ./videos_pid/")

"""* General description

  * This script defines two target-generating functions (static point and circular motion), implements a Cartesian-space PID controller that converts end-effector error into joint torques via a Jacobian-transpose mapping, evaluates performance with several tracking/control metrics, and runs two rollout experiments in the `TwoLinkReacherEnv` (static target and moving circular target).
  * Optionally, it records rollout videos (via `RecordVideo`) and displays the saved MP4s in a notebook environment.

* Target functions

  * `static_target_fn(t)`

    * Returns a constant target position `(1.5, 0.0)` for all times `t` (static goal).
  * `circular_target_fn(t)`

    * Returns a target moving on a circle centered at the origin with radius `0.5` and angular speed `ω = 2π/15`, i.e., `(r cos(ωt), r sin(ωt))`.

* `CartesianPIDController` (Cartesian PID + Jacobian-transpose torque mapping)

  * `__init__(l1, l2, kp, ki, kd, dt, torque_limit, integral_clip)`

    * Stores arm geometry, constructs diagonal gain matrices `Kp/Ki/Kd`, sets timestep and torque saturation, sets integral anti-windup clip bounds, and calls `reset()` to initialize controller state.
  * `reset()`

    * Clears the integral accumulator and previous-error memory; marks the next control call as the first step so the derivative term can be initialized safely.
  * `forward_kinematics(q)`

    * Computes end-effector Cartesian position from joint angles `(θ1, θ2)` for a planar 2-link arm.
  * `jacobian(q)`

    * Computes the 2×2 geometric Jacobian `J(q)` mapping joint velocities to end-effector linear velocity in Cartesian space.
  * `act(q, dq, p_target) -> (tau, info)`

    * Core control law:

      * Computes end-effector position `p_end` and Cartesian error `e = p_target - p_end`.
      * Updates integral term `∫e dt` with clipping (`integral_clip`) for anti-windup.
      * Approximates derivative term `de/dt` using finite differences (zeroed on the first call).
      * Forms a Cartesian “force-like” command `u = Kp e + Ki ∫e dt + Kd de/dt`.
      * Maps to joint torques via Jacobian transpose: `tau = J(q)^T u`.
      * Clips `tau` to actuator limits and returns it along with diagnostic info (positions, error vector/norm, `u`, and `J`).
    * Note: `dq` is passed in but not used directly; the derivative term is computed from error differences, not from `dq` or `J dq`.

* Evaluation metrics

  * `compute_metrics(errors, taus, epsilon=0.05)`

    * Aggregates rollout traces into required metrics:

      * Mean tracking error: average of the per-step error magnitudes.
      * Success rate: fraction of timesteps where error `< epsilon`.
      * Control energy: sum over time of `||tau||²` (squared torque magnitude).
      * Smoothness: sum over time of `||Δtau||²` (squared step-to-step torque changes), zero if fewer than 2 actions.

* Rollout / experiment runner

  * `run_pid_episode(target_mode, video_folder=None, video_name="pid_rollout", render_video=False, seed=0, max_steps=500, dt=0.02, torque_limit=2.0, epsilon_success=0.05, gains=None)`

    * Creates a `TwoLinkReacherEnv` configured for the chosen target (callable `target_mode`), timestep, horizon, torque limits, and random initial state.
    * Optionally wraps the env with `RecordVideo` when `render_video=True` and a `video_folder` is provided (render mode becomes `rgb_array`).
    * Instantiates `CartesianPIDController` with provided gains and runs a full episode loop until termination/truncation:

      * Reads true state from `env.unwrapped` (`q`, `dq`, `target`).
      * Computes torques from the controller and steps the environment with `env.step(tau)`.
      * Logs error norms, torques, end-effector positions, and targets.
    * Computes summary metrics via `compute_metrics(...)` and returns `(metrics, traces)` where traces include time series for error, torque, end-effector position, and target position.

* Experiment configuration and execution

  * `gains = {"kp": (14,14), "ki": (0.2,0.2), "kd": (10,10)}`

    * Sets PID gains used in both scenarios.
  * Static scenario call: `run_pid_episode(target_mode=static_target_fn, ..., epsilon_success=0.15)`

    * Runs a rollout tracking a fixed target at `(1.5, 0.0)`, records a video named `pid_static-episode-0.mp4` into `videos_pid/`.
  * Circular scenario call: `run_pid_episode(target_mode=circular_target_fn, ..., epsilon_success=0.15)`

    * Runs a rollout tracking a moving circular target, records `pid_circle-episode-0.mp4` into `videos_pid/`.
  * Printing and display block

    * Prints metrics for each scenario and displays the saved MP4s inside the notebook (via `display(Video(...))`), and prints the output directory path.

# Part (c): Joint-Space PID + Inverse Kinematics
"""

