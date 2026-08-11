
import math
import numpy as np
from gymnasium.wrappers import RecordVideo

# ============================================================
# Inverse Kinematics
# ============================================================

def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def ik_two_link(
    x: float,
    y: float,
    l1: float = 1.0,
    l2: float = 1.0,
    elbow: str = "down",
) -> np.ndarray:
    """
    Analytic IK for 2-link planar arm.

    Returns:
        qd = [theta1_d, theta2_d]

    elbow:
        "down" -> +sqrt branch
        "up"   -> -sqrt branch
    """
    r2 = x * x + y * y

    c2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    c2 = np.clip(c2, -1.0, 1.0)

    s2_abs = math.sqrt(max(0.0, 1.0 - c2 * c2))
    s2 = s2_abs if elbow == "down" else -s2_abs

    theta2 = math.atan2(s2, c2)

    alpha = math.atan2(y, x)
    beta = math.atan2(l2 * s2, l1 + l2 * c2)
    theta1 = alpha - beta

    return np.array([wrap_pi(theta1), wrap_pi(theta2)], dtype=np.float64)


# ============================================================
# Joint-space PID controller
# ============================================================

class JointPIDIKController:
    """
    1) Compute q_d = IK(x_t, y_t)
    2) Track q_d using independent joint PIDs:
           tau_i = Kp_i e_i + Ki_i ∫e_i dt + Kd_i de_i/dt
    """

    def __init__(
        self,
        l1=1.0,
        l2=1.0,
        dt=0.02,
        torque_limit=2.0,
        elbow="down",
        kp=(18.0, 14.0),
        ki=(0.4, 0.3),
        kd=(5.0, 4.0),
        integral_clip=(2.0, 2.0),
    ):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.dt = float(dt)
        self.torque_limit = float(torque_limit)
        self.elbow = elbow

        self.kp = np.asarray(kp, dtype=np.float64)
        self.ki = np.asarray(ki, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.integral_clip = np.asarray(integral_clip, dtype=np.float64)

        self.reset()

    def reset(self):
        self.e_int = np.zeros(2, dtype=np.float64)
        self.e_prev = np.zeros(2, dtype=np.float64)
        self.first = True
        self.qd_prev = None

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        th1, th2 = q
        x = self.l1 * np.cos(th1) + self.l2 * np.cos(th1 + th2)
        y = self.l1 * np.sin(th1) + self.l2 * np.sin(th1 + th2)
        return np.array([x, y], dtype=np.float64)

    def compute_desired_joints(self, p_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """
        IK with simple branch continuity:
        - compute both elbow-down and elbow-up
        - choose the one closer to current q
        (unless self.elbow is explicitly fixed)
        """
        x, y = float(p_target[0]), float(p_target[1])

        if self.elbow in ("down", "up"):
            qd = ik_two_link(x, y, self.l1, self.l2, elbow=self.elbow)
            return qd

        # Optional automatic branch selection
        qd_down = ik_two_link(x, y, self.l1, self.l2, elbow="down")
        qd_up = ik_two_link(x, y, self.l1, self.l2, elbow="up")

        e_down = np.array([wrap_pi(qd_down[0] - q_current[0]), wrap_pi(qd_down[1] - q_current[1])])
        e_up = np.array([wrap_pi(qd_up[0] - q_current[0]), wrap_pi(qd_up[1] - q_current[1])])

        if np.linalg.norm(e_down) <= np.linalg.norm(e_up):
            return qd_down
        return qd_up

    def act(self, q: np.ndarray, dq: np.ndarray, p_target: np.ndarray) -> tuple[np.ndarray, dict]:
        # IK -> desired joints
        qd = self.compute_desired_joints(p_target, q)

        # Joint error (angle-wrapped)
        e = np.array([
            wrap_pi(qd[0] - q[0]),
            wrap_pi(qd[1] - q[1]),
        ], dtype=np.float64)

        # Integral term (anti-windup clip)
        self.e_int += e * self.dt
        self.e_int = np.clip(self.e_int, -self.integral_clip, self.integral_clip)

        # Derivative term
        if self.first:
            de = np.zeros(2, dtype=np.float64)
            self.first = False
        else:
            de = (e - self.e_prev) / self.dt
        self.e_prev = e.copy()

        # Joint PID torques
        tau = self.kp * e + self.ki * self.e_int + self.kd * de
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)

        # End-effector tracking error for evaluation (project metric)
        p_end = self.forward_kinematics(q)
        e_cart = p_target - p_end

        info = {
            "qd": qd,
            "q": q.copy(),
            "dq": dq.copy(),
            "e_joint": e,
            "de_joint": de,
            "p_end": p_end,
            "p_target": p_target.copy(),
            "e_cart_vec": e_cart,
            "e_cart_norm": float(np.linalg.norm(e_cart)),
        }
        return tau, info

# ============================================================
# Rollout runner
# ============================================================

def run_joint_pid_ik_episode(
    target_mode,
    video_folder: str | None = None,
    video_name: str = "joint_pid_ik",
    render_video: bool = False,
    seed: int = 0,
    max_steps: int = 150,
    dt: float = 0.02,
    torque_limit: float = 2.0,
    epsilon_success: float = 0.05,
    gains=None,
    elbow="down",
):
    if gains is None:
        gains = {
            "kp": (18.0, 14.0),
            "ki": (0.4, 0.3),
            "kd": (5.0, 4.0),
        }

    render_mode = "rgb_array" if render_video else None

    env = TwoLinkReacherEnv(
        dt=dt,
        render_mode=render_mode,
        obs_mode="minimal",
        max_steps=max_steps,
        torque_limit=torque_limit,
        target_mode=target_mode,
        success_radius=None,         # track full horizon
        reward_torque_penalty=0.0,
        draw_trail=True,
        trail_len=80,
        # random_init=True
    )

    if render_video and video_folder is not None:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep: True,
            name_prefix=video_name,
        )

    base_env = env.unwrapped

    ctrl = JointPIDIKController(
        l1=1.0,
        l2=1.0,
        dt=dt,
        torque_limit=torque_limit,
        elbow=elbow,
        kp=gains["kp"],
        ki=gains["ki"],
        kd=gains["kd"],
        integral_clip=(2.0, 2.0),
    )

    obs, info = env.reset(seed=seed)
    ctrl.reset()

    errors = []
    taus = []
    p_ends = []
    p_targets = []
    qds = []

    terminated = truncated = False
    while not (terminated or truncated):
        q = base_env.q.copy()
        dq = base_env.dq.copy()

        t_now = base_env.steps * base_env.dt
        p_target = np.asarray(base_env.target_fn(t_now), dtype=np.float64)

        tau, cinfo = ctrl.act(q=q, dq=dq, p_target=p_target)

        obs, reward, terminated, truncated, info = env.step(tau)

        errors.append(cinfo["e_cart_norm"])
        taus.append(tau.copy())
        p_ends.append(cinfo["p_end"].copy())
        p_targets.append(cinfo["p_target"].copy())
        qds.append(cinfo["qd"].copy())

    env.close()

    metrics = compute_metrics(errors, taus, epsilon=epsilon_success)
    traces = {
        "errors": np.asarray(errors),
        "taus": np.asarray(taus),
        "p_end": np.asarray(p_ends),
        "p_target": np.asarray(p_targets),
        "qd": np.asarray(qds),
    }
    return metrics, traces

gains = {
    "kp": (10.0, 8.0),
    "ki": (0.0, 0.0),
    "kd": (3.5, 3.0),
}

static_metrics, _ = run_joint_pid_ik_episode(
    target_mode=static_target_fn,
    render_video=True,
    video_folder="videos_pid_ik",
    video_name="pidik_static",
    seed=42,
    gains=gains,
    elbow="down",
    epsilon_success=0.05,
)

circ_metrics, _ = run_joint_pid_ik_episode(
    target_mode=circular_target_fn,
    render_video=True,
    video_folder="videos_pid_ik",
    video_name="pidik_circle",
    seed=42,
    gains=gains,
    elbow="down",
    epsilon_success=0.05,
)

print("\n=== Joint PID + IK Results ===")
print("Gains:", gains)

print("\nStatic target (1.5, 0):")
for k, v in static_metrics.items():
    print(f"  {k}: {v:.6f}")

video_dir = "videos_pid_ik"
video_path = os.path.join(video_dir, "pidik_static-episode-0.mp4")
print(f"\nDisplaying: {video_path}")
display(Video(video_path, embed=True))

print("\nCircular target (r=0.5, center=(0,0)):")
for k, v in circ_metrics.items():
    print(f"  {k}: {v:.6f}")

video_path = os.path.join(video_dir, "pidik_circle-episode-0.mp4")
print(f"\nDisplaying: {video_path}")
display(Video(video_path, embed=True))

print("\nVideos saved in ./videos_pid_ik/")

"""* General description

  * This script implements an analytic inverse-kinematics (IK) solver for a 2-link planar arm and a joint-space PID controller that tracks IK-computed desired joint angles to make the end-effector follow a Cartesian target.
  * It then runs two rollouts in `TwoLinkReacherEnv` (static target and circular moving target), computes standard tracking/control metrics using `compute_metrics`, optionally records videos via `RecordVideo`, and prints/displays the results in a notebook.

* Inverse kinematics utilities

  * `wrap_pi(angle)`

    * Wraps any angle to the interval `[-π, π)` to avoid discontinuities when comparing/controlling angles.
  * `ik_two_link(x, y, l1=1.0, l2=1.0, elbow="down")`

    * Analytic IK for a planar 2-link manipulator that returns desired joint angles `qd = [θ1_d, θ2_d]` for a desired end-effector point `(x, y)`.
    * Computes `cos(θ2)` from the law of cosines, clips to `[-1, 1]` for numerical safety, selects the elbow branch (`"down"` uses `+sqrt`, `"up"` uses `-sqrt`), then solves `θ1` via geometric decomposition (`alpha - beta`).
    * Wraps both angles with `wrap_pi` before returning.

* `JointPIDIKController` (joint PID tracking an IK reference)

  * `__init__(l1, l2, dt, torque_limit, elbow, kp, ki, kd, integral_clip)`

    * Stores geometry and controller parameters, converts gains/clips to numpy arrays, sets torque saturation, chooses IK elbow policy, and initializes internal PID state via `reset()`.
  * `reset()`

    * Resets integral accumulator, previous error, and first-step flag; clears any stored previous desired joint state (`qd_prev` is present but not used elsewhere in this code).
  * `forward_kinematics(q)`

    * Computes end-effector position `(x, y)` from current joint angles; used for logging the Cartesian tracking error.
  * `compute_desired_joints(p_target, q_current)`

    * Computes the desired joint angles `qd` from the Cartesian target using IK.
    * If `self.elbow` is `"down"` or `"up"`, it uses that fixed branch.
    * Otherwise, it computes both elbow branches and selects the one closer to the current joint configuration using wrapped joint differences (a simple branch-continuity heuristic to reduce sudden IK jumps).
  * `act(q, dq, p_target) -> (tau, info)`

    * Main control step:

      * Uses IK to compute `qd` for the current Cartesian target.
      * Computes wrapped joint error `e = wrap_pi(qd - q)` per joint.
      * Updates integral term `∫e dt` with per-joint anti-windup clipping (`integral_clip`).
      * Computes derivative term via finite differences of the joint error (zero on the first call).
      * Forms joint torques with independent joint PIDs: `tau = kp*e + ki*e_int + kd*de`, then clips to `[-torque_limit, torque_limit]`.
      * Computes end-effector Cartesian error `e_cart = p_target - p_end` for evaluation/logging.
    * Returns `(tau, info)` where `info` contains desired joints `qd`, joint errors, end-effector position, target, and Cartesian error norm.

* Rollout runner

  * `run_joint_pid_ik_episode(target_mode, video_folder=None, video_name="joint_pid_ik", render_video=False, seed=0, max_steps=500, dt=0.02, torque_limit=2.0, epsilon_success=0.05, gains=None, elbow="down")`

    * Creates a `TwoLinkReacherEnv` configured with the requested target generator (`target_mode` callable), horizon, timestep, and torque limits; uses `rgb_array` rendering only if video recording is enabled.
    * Optionally wraps the environment with `RecordVideo` to save rollouts as MP4.
    * Instantiates `JointPIDIKController` with the specified gains and elbow policy, resets env and controller, then iterates until termination/truncation:

      * Reads `q` and `dq` from `env.unwrapped`.
      * Computes the current time `t_now = steps * dt` and evaluates the target explicitly via `base_env.target_fn(t_now)` (instead of using `base_env.target`), ensuring the controller uses the same time-parameterized trajectory.
      * Calls `ctrl.act(...)` to get torques and steps the environment with `env.step(tau)`.
      * Logs Cartesian error norm, torques, end-effector positions, targets, and desired joints.
    * Computes summary metrics with `compute_metrics(errors, taus, epsilon=epsilon_success)` and returns `(metrics, traces)`.

* Experiment configuration and execution

  * `gains = {"kp": (10, 8), "ki": (0, 0), "kd": (3.5, 3.0)}`

    * Sets joint PID gains for both scenarios (note integral is disabled here).
  * Static scenario

    * Runs `run_joint_pid_ik_episode(target_mode=static_target_fn, ...)`, records `pidik_static-episode-0.mp4` into `videos_pid_ik/`, prints the metric dictionary, and displays the video in-notebook.
  * Circular scenario

    * Runs `run_joint_pid_ik_episode(target_mode=circular_target_fn, ...)`, records `pidik_circle-episode-0.mp4` into `videos_pid_ik/`, prints metrics, and displays the video.
  * Output section

    * Prints a summary header, prints metrics for each run, shows the video paths being displayed, and notes the directory where videos are saved.

# Part (d): Discrete Action Control with N-Step SARSA
"""