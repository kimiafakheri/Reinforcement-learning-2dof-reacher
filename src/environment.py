# -*- coding: utf-8 -*-
"""
# Reinforcement Learning Term Project: Two-Link Reacher (Control + RL)
---
Electrical Engineering Department - Sharif Universtiy of Technology - Kimia Fakheri

# Part (a): Implementation of the Environment (Gymnasium)
"""

import math
from typing import Optional, Tuple, Callable, List, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
except ImportError:
    pygame = None


# ============================================================
# Target trajectory factory functions
# ============================================================

def target_static_fixed(x: float, y: float) -> Callable[[float], np.ndarray]:
    """Always returns the same target (x, y)."""
    p = np.array([x, y], dtype=np.float64)

    def _f(t: float) -> np.ndarray:
        return p.copy()

    return _f


def target_static_random_factory(
    rng,
    l1: float = 1.0,
    l2: float = 1.0,
    min_r: float = 0.2,
    margin: float = 0.1,
) -> Callable[[], Callable[[float], np.ndarray]]:
    """
    Returns a factory that samples ONE random target per episode,
    then returns a constant function f(t) = p_episode.
    """
    max_r = l1 + l2 - margin

    def _make_episode_fn() -> Callable[[float], np.ndarray]:
        # Uniform in disc
        r = max_r * np.sqrt(rng.uniform(0.0, 1.0))
        r = max(r, min_r)
        phi = rng.uniform(-np.pi, np.pi)
        p = np.array([r * np.cos(phi), r * np.sin(phi)], dtype=np.float64)

        def _f(t: float) -> np.ndarray:
            return p.copy()

        return _f

    return _make_episode_fn


def target_circular_motion(
    xc: float = 0.0,
    yc: float = 0.0,
    r: float = 0.5,
    omega: float = 2.0 * np.pi / 15.0,
    phase: float = 0.0,
) -> Callable[[float], np.ndarray]:
    """
    x(t) = xc + r cos(ω t + phase)
    y(t) = yc + r sin(ω t + phase)
    """
    c = np.array([xc, yc], dtype=np.float64)

    def _f(t: float) -> np.ndarray:
        return c + r * np.array(
            [np.cos(omega * t + phase), np.sin(omega * t + phase)],
            dtype=np.float64,
        )

    return _f


def target_sine_pair() -> Callable[[float], np.ndarray]:
    """
    x(t) = sin(2t + π/2)
    y(t) = sin(4t)
    """
    def _f(t: float) -> np.ndarray:
        return np.array(
            [np.sin(2.0 * t + np.pi / 2.0), np.sin(4.0 * t)],
            dtype=np.float64,
        )
    return _f


class TwoLinkReacherEnv(gym.Env):
    """
    Enhanced 2-link planar reacher (Gymnasium-style) with configurable target trajectories.

    Dynamics:
        M(q) qdd + C(q,dq)dq + B dq = tau
    Discretization:
        semi-implicit Euler

    Observation modes:
        - "minimal"      : (xt, yt, th1, th2, dth1, dth2)
        - "minimal_ee"   : (xt, yt, th1, th2, dth1, dth2, xend, yend)
        - "error_only"   : (xt-xend, yt-yend, th1, th2, dth1, dth2)
        - "ik_error"     : (th1-th1_target, th2-th2_target, dth1, dth2)

    Target modes:
        - "static_fixed"  (requires target=(x,y))
        - "static_random" (sampled once per episode)
        - "circle_origin" (r=0.5, center=(0,0), omega=2π/15)
        - "lissajous"     (x=sin(2t+π/2), y=sin(4t))
        - callable f(t)->np.ndarray shape (2,)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        dt: float = 0.02,
        target: Optional[Tuple[float, float]] = None,
        render_mode: Optional[str] = None,
        obs_mode: str = "minimal",
        max_steps: int = 150,
        torque_limit: float = 2.0,
        wrap_angles: bool = True,
        seed: Optional[int] = None,
        target_mode: Union[str, Callable[[float], np.ndarray]] = "static_random",
        success_radius: Optional[float] = None,
        reward_torque_penalty: float = 0.0,
        draw_trail: bool = True,
        trail_len: int = 60,
        random_init=False
    ):
        super().__init__()

        # ---------------------------
        # Physical parameters (given)
        # ---------------------------
        self.l1 = 1.0
        self.l2 = 1.0
        self.m1 = 1.0
        self.m2 = 1.0
        self.b1 = 0.05
        self.b2 = 0.05

        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.torque_limit = float(torque_limit)
        self.wrap_angles = bool(wrap_angles)

        self.random_init = random_init

        # Fancy config
        self.success_radius = success_radius
        self.reward_torque_penalty = float(reward_torque_penalty)
        self.draw_trail = bool(draw_trail)
        self.trail_len = int(trail_len)
        self._trail: List[np.ndarray] = []

        # Render config
        assert render_mode in (None, "human", "rgb_array")
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.window_size = 700
        self._font = None

        # Observation mode
        valid_obs_modes = {"minimal", "minimal_ee", "error_only", "ik_error"}
        if obs_mode not in valid_obs_modes:
            raise ValueError(f"obs_mode must be one of {valid_obs_modes}, got {obs_mode}")
        self.obs_mode = obs_mode

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Target trajectory config
        self.target_mode = target_mode
        self.target_fn: Optional[Callable[[float], np.ndarray]] = None
        self.fixed_target = None if target is None else np.array(target, dtype=np.float64)

        # If target is explicitly provided, default to static_fixed unless callable given
        if self.fixed_target is not None and isinstance(self.target_mode, str):
            if self.target_mode in ("static_random", "static_fixed"):
                self.target_mode = "static_fixed"

        # State
        self.q = np.zeros(2, dtype=np.float64)
        self.dq = np.zeros(2, dtype=np.float64)
        self.target = np.zeros(2, dtype=np.float64)
        self.steps = 0

        # Dynamics constants (as specified)
        self.a = self.m1 * self.l1**2 + self.m2 * (self.l1**2 + self.l2**2)
        self.b = self.m2 * self.l1 * self.l2
        self.d = self.m2 * self.l2**2

        # Spaces
        self.action_space = spaces.Box(
            low=-self.torque_limit,
            high=self.torque_limit,
            shape=(2,),
            dtype=np.float32,
        )

        obs_dim = self._obs_dim()
        high = np.full((obs_dim,), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Math helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _wrap_q(self):
        self.q[0] = self._wrap_pi(self.q[0])
        self.q[1] = self._wrap_pi(self.q[1])

    def _obs_dim(self) -> int:
        return {
            "minimal": 6,
            "minimal_ee": 8,
            "error_only": 6,
            "ik_error": 4,
        }[self.obs_mode]

    def _forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        th1, th2 = q
        x = self.l1 * np.cos(th1) + self.l2 * np.cos(th1 + th2)
        y = self.l1 * np.sin(th1) + self.l2 * np.sin(th1 + th2)
        return np.array([x, y], dtype=np.float64)

    def _inverse_kinematics(self, target_xy: np.ndarray) -> np.ndarray:
        """
        One analytic IK branch (elbow-down by convention).
        """
        x, y = float(target_xy[0]), float(target_xy[1])
        r2 = x * x + y * y
        c2 = (r2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        c2 = np.clip(c2, -1.0, 1.0)
        s2 = np.sqrt(max(0.0, 1.0 - c2 * c2))
        th2 = np.arctan2(s2, c2)

        k1 = self.l1 + self.l2 * c2
        k2 = self.l2 * s2
        th1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        return np.array([self._wrap_pi(th1), self._wrap_pi(th2)], dtype=np.float64)

    def _distance(self) -> float:
        p_end = self._forward_kinematics(self.q)
        return float(np.linalg.norm(p_end - self.target))

    def _loss(self) -> float:
        p_end = self._forward_kinematics(self.q)
        e = p_end - self.target
        return float(e @ e)

    def _setup_target_function_for_episode(self):
        """
        Creates self.target_fn for the current episode.
        Supports:
          - string presets
          - custom callable f(t)->(x,y)
        """
        # Custom callable passed directly
        if callable(self.target_mode):
            self.target_fn = self.target_mode
            return

        mode = self.target_mode

        if mode == "static_fixed":
            if self.fixed_target is None:
                raise ValueError("target_mode='static_fixed' requires `target=(x,y)`.")
            self.target_fn = target_static_fixed(float(self.fixed_target[0]), float(self.fixed_target[1]))

        elif mode == "static_random":
            make_fn = target_static_random_factory(
                rng=self.np_random,
                l1=self.l1,
                l2=self.l2,
                min_r=0.2,
                margin=0.1,
            )
            self.target_fn = make_fn()

        elif mode == "circle_origin":
            # Required circular trajectory:
            # r=0.5, center=(0,0), omega=(2π/3)*0.2 = 2π/15
            self.target_fn = target_circular_motion(
                xc=0.0,
                yc=0.0,
                r=0.5,
                omega=2.0 * np.pi / 15.0,
            )

        elif mode == "lissajous":
            # x(t)=sin(2t+π/2), y(t)=sin(4t)
            self.target_fn = target_sine_pair()

        else:
            raise ValueError(
                "Unknown target_mode. Use one of "
                "{'static_fixed','static_random','circle_origin','lissajous'} "
                "or pass a callable f(t)->(x,y)."
            )

    def _update_target(self):
        if self.target_fn is None:
            raise RuntimeError("target_fn not initialized. Call _setup_target_function_for_episode() in reset().")
        t = self.steps * self.dt
        self.target = np.asarray(self.target_fn(t), dtype=np.float64).reshape(2,)

    def _get_obs(self) -> np.ndarray:
        th1, th2 = self.q
        dth1, dth2 = self.dq
        xt, yt = self.target
        xend, yend = self._forward_kinematics(self.q)

        if self.obs_mode == "minimal":
            obs = np.array([xt, yt, th1, th2, dth1, dth2], dtype=np.float32)
        elif self.obs_mode == "minimal_ee":
            obs = np.array([xt, yt, th1, th2, dth1, dth2, xend, yend], dtype=np.float32)
        elif self.obs_mode == "error_only":
            obs = np.array([xt - xend, yt - yend, th1, th2, dth1, dth2], dtype=np.float32)
        elif self.obs_mode == "ik_error":
            q_t = self._inverse_kinematics(self.target)
            err = self.q - q_t
            err = np.array([self._wrap_pi(err[0]), self._wrap_pi(err[1])], dtype=np.float64)
            obs = np.array([err[0], err[1], dth1, dth2], dtype=np.float32)
        else:
            raise RuntimeError("Unknown obs_mode")

        return obs

    # -------------------------------------------------------------------------
    # Gym API
    # -------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self._trail = []

        # Initial pose/velocity
        if self.random_init:
            self.q = self.np_random.uniform(low=-0.6, high=0.6, size=(2,)).astype(np.float64)
            self.dq = self.np_random.uniform(low=-0.15, high=0.15, size=(2,)).astype(np.float64)
        else:
            self.q = np.zeros((2,), dtype=np.float64)
            self.dq = np.zeros((2,), dtype=np.float64)

        if self.wrap_angles:
            self._wrap_q()

        # Configure target trajectory for this episode and evaluate at t=0
        self._setup_target_function_for_episode()
        self._update_target()

        self._trail.append(self._forward_kinematics(self.q).copy())

        obs = self._get_obs()
        info = self._build_info(success=False)

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(2,)
        tau = np.clip(action, -self.torque_limit, self.torque_limit)

        # Update target using current time t = steps * dt (before dynamics)
        self._update_target()

        th1, th2 = self.q
        dth1, dth2 = self.dq

        # Inertia matrix M(q)
        c2 = np.cos(th2)
        M = np.array(
            [
                [self.a + 2 * self.b * c2, self.d + self.b * c2],
                [self.d + self.b * c2, self.d],
            ],
            dtype=np.float64,
        )

        # Coriolis/Centrifugal vector C(q,dq)dq
        s2 = np.sin(th2)
        Cqdq = np.array(
            [
                -self.b * s2 * (2 * dth1 * dth2 + dth2**2),
                self.b * s2 * (dth1**2),
            ],
            dtype=np.float64,
        )

        # Damping
        Bdq = np.array([self.b1 * dth1, self.b2 * dth2], dtype=np.float64)

        # Solve M qdd = tau - Cqdq - Bdq
        rhs = tau - Cqdq - Bdq
        qdd = np.linalg.solve(M, rhs)

        # Semi-implicit Euler
        self.dq = self.dq + qdd * self.dt
        self.q = self.q + self.dq * self.dt
        if self.wrap_angles:
            self._wrap_q()

        self.steps += 1

        # Trail
        if self.draw_trail:
            self._trail.append(self._forward_kinematics(self.q).copy())
            if len(self._trail) > self.trail_len:
                self._trail.pop(0)

        # Reward (project reward + optional torque penalty)
        loss = self._loss()
        reward = -loss - self.reward_torque_penalty * float(tau @ tau)

        dist = math.sqrt(loss)
        success = (self.success_radius is not None) and (dist <= self.success_radius)

        terminated = bool(success)
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = self._build_info(success=success, tau=tau, qdd=qdd)

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _build_info(self, success: bool, tau=None, qdd=None):
        p_end = self._forward_kinematics(self.q)
        loss = self._loss()
        info = {
            "target": self.target.copy(),
            "end_effector": p_end.copy(),
            "loss": float(loss),
            "distance": float(np.sqrt(loss)),
            "success": bool(success),
            "q": self.q.copy(),
            "dq": self.dq.copy(),
        }
        if tau is not None:
            info["tau"] = np.asarray(tau, dtype=np.float64).copy()
        if qdd is not None:
            info["qdd"] = np.asarray(qdd, dtype=np.float64).copy()
        return info

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------
    def _init_pygame(self):
        if pygame is None:
            raise ImportError("pygame is required for rendering. Install with: pip install pygame")

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Two-Link Reacher")
            else:
                self.screen = pygame.Surface((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self._font = pygame.font.SysFont("consolas", 16)

    def _world_to_screen(self, p: np.ndarray) -> Tuple[int, int]:
        # World range approx [-2.4, 2.4]
        scale = self.window_size / 5.2
        x = int(self.window_size * 0.5 + p[0] * scale)
        y = int(self.window_size * 0.5 - p[1] * scale)
        return x, y

    def _draw_grid(self, surf):
        # light grid
        for v in np.arange(-2.0, 2.01, 0.5):
            p1 = self._world_to_screen(np.array([-2.4, v]))
            p2 = self._world_to_screen(np.array([2.4, v]))
            pygame.draw.line(surf, (238, 238, 238), p1, p2, 1)

            p3 = self._world_to_screen(np.array([v, -2.4]))
            p4 = self._world_to_screen(np.array([v, 2.4]))
            pygame.draw.line(surf, (238, 238, 238), p3, p4, 1)

        # axes
        cx, cy = self._world_to_screen(np.array([0.0, 0.0]))
        pygame.draw.line(surf, (210, 210, 210), (0, cy), (self.window_size, cy), 2)
        pygame.draw.line(surf, (210, 210, 210), (cx, 0), (cx, self.window_size), 2)

    def render(self):
        if self.render_mode is None:
            return None

        self._init_pygame()
        canvas = self.screen

        # Event handling (important for human mode)
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        # Background + grid
        canvas.fill((250, 250, 252))
        self._draw_grid(canvas)

        # Workspace circles
        origin_px = self._world_to_screen(np.array([0.0, 0.0]))
        scale = self.window_size / 5.2
        outer_r = int((self.l1 + self.l2) * scale)
        inner_r = int(abs(self.l1 - self.l2) * scale)
        pygame.draw.circle(canvas, (225, 225, 228), origin_px, outer_r, 2)
        if inner_r > 0:
            pygame.draw.circle(canvas, (232, 232, 236), origin_px, inner_r, 1)

        # Kinematics points
        th1, th2 = self.q
        p0 = np.array([0.0, 0.0], dtype=np.float64)
        p1 = np.array([self.l1 * np.cos(th1), self.l1 * np.sin(th1)], dtype=np.float64)
        p2 = self._forward_kinematics(self.q)
        pt = self.target

        p0s = self._world_to_screen(p0)
        p1s = self._world_to_screen(p1)
        p2s = self._world_to_screen(p2)
        pts = self._world_to_screen(pt)

        # Trail
        if self.draw_trail and len(self._trail) > 1:
            n = len(self._trail)
            for i in range(1, n):
                a = self._world_to_screen(self._trail[i - 1])
                b = self._world_to_screen(self._trail[i])
                width = 1 if i < n * 0.6 else 2
                pygame.draw.line(canvas, (140, 200, 255), a, b, width)

        # Target marker
        pygame.draw.circle(canvas, (220, 35, 35), pts, 8)
        pygame.draw.circle(canvas, (220, 35, 35), pts, 16, 2)
        pygame.draw.line(canvas, (220, 35, 35), (pts[0] - 12, pts[1]), (pts[0] + 12, pts[1]), 2)
        pygame.draw.line(canvas, (220, 35, 35), (pts[0], pts[1] - 12), (pts[0], pts[1] + 12), 2)

        # Success radius visual
        if self.success_radius is not None:
            sr_px = max(2, int(self.success_radius * scale))
            pygame.draw.circle(canvas, (255, 180, 180), pts, sr_px, 1)

        # Draw links
        pygame.draw.line(canvas, (70, 90, 180), p0s, p1s, 10)
        pygame.draw.line(canvas, (40, 60, 140), p0s, p1s, 4)

        pygame.draw.line(canvas, (70, 170, 120), p1s, p2s, 10)
        pygame.draw.line(canvas, (35, 120, 80), p1s, p2s, 4)

        # Joints and EE
        pygame.draw.circle(canvas, (35, 35, 35), p0s, 8)
        pygame.draw.circle(canvas, (35, 35, 35), p1s, 8)
        pygame.draw.circle(canvas, (20, 145, 40), p2s, 9)

        # HUD
        loss = self._loss()
        dist = np.sqrt(loss)
        t_now = self.steps * self.dt
        lines = [
            f"step: {self.steps}/{self.max_steps}",
            f"time: {t_now:.2f} s",
            f"dist: {dist:.4f}",
            f"reward: {-loss:.4f}" if self.reward_torque_penalty == 0 else f"reward~: {-loss:.4f} - λ||τ||²",
            f"q:  [{self.q[0]:+6.3f}, {self.q[1]:+6.3f}]",
            f"dq: [{self.dq[0]:+6.3f}, {self.dq[1]:+6.3f}]",
            f"target_mode: {self.target_mode if isinstance(self.target_mode, str) else 'callable'}",
            f"obs_mode: {self.obs_mode}",
        ]
        x_text, y_text = 12, 10
        for line in lines:
            txt = self._font.render(line, True, (25, 25, 25))
            canvas.blit(txt, (x_text, y_text))
            y_text += 19

        if self.render_mode == "human":
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(int(round(1.0 / self.dt)))
            return None

        # rgb_array mode
        rgb = pygame.surfarray.array3d(canvas)   # (W,H,3)
        rgb = np.transpose(rgb, (1, 0, 2))       # -> (H,W,3)
        return rgb

    def close(self):
        if pygame is not None and self.screen is not None:
            pygame.quit()
        self.screen = None
        self.clock = None
        self._font = None

# Example Usages:
# # random target per episode
# env = TwoLinkReacherEnv(target_mode="static_random", render_mode="human")

# # circular target around origin
# env = TwoLinkReacherEnv(target_mode="circle_origin", render_mode="human")

# # x(t)=sin(2t+pi/2), y(t)=sin(4t)
# env = TwoLinkReacherEnv(target_mode="lissajous", render_mode="human")

# # custom callable
# def my_target(t):
#     return np.array([0.6*np.cos(t), 0.3*np.sin(3*t)], dtype=np.float64)

# env = TwoLinkReacherEnv(target_mode=my_target, render_mode="human")

"""### Throughout this notebook, I use a fixed simulation and control setup: the environment time step is $dt = 0.02$ s, each joint torque is bounded to $\tau \in [-2, 2]$, and each episode is truncated at $\text{max\_steps} = 150$.

* General description

  * This script defines a custom Gymnasium environment for a 2-link planar robotic arm (“reacher”) that must move its end-effector to a target point in 2D.
  * The target can be fixed, randomly sampled per episode, follow built-in time trajectories (circle, Lissajous), or be provided as a user callable `f(t) -> (x, y)`.
  * The environment simulates simple rigid-body joint dynamics with damping, integrates them forward in time with semi-implicit Euler, emits configurable observation vectors, computes a distance-based reward (with optional torque penalty), and optionally renders the scene using pygame (human window or RGB array frames).

* Target trajectory factory functions

  * `target_static_fixed(x, y)`

    * Returns a function `f(t)` that always outputs the same `(x, y)` target for all times `t`.
  * `target_static_random_factory(rng, l1=1.0, l2=1.0, min_r=0.2, margin=0.1)`

    * Returns a *factory* that, when called at the start of an episode, samples one random reachable target point (uniform over a disc, clamped to at least `min_r`) and returns a constant function `f(t)` for that episode.
  * `target_circular_motion(xc=0.0, yc=0.0, r=0.5, omega=2π/15, phase=0.0)`

    * Returns a function `f(t)` that moves the target in a circle of radius `r` around center `(xc, yc)` with angular speed `omega` and phase offset `phase`.
  * `target_sine_pair()`

    * Returns a function `f(t)` implementing a 2D Lissajous-style motion: `x(t)=sin(2t+π/2)`, `y(t)=sin(4t)`.

* `TwoLinkReacherEnv` (Gymnasium environment)

  * `__init__(...)`

    * Sets physical parameters (link lengths, masses, damping), simulation parameters (`dt`, `max_steps`, torque limits, angle wrapping), RNG seeding, target configuration (`target_mode`, optional fixed `target`), observation mode selection, action/observation spaces, and rendering/trail settings.
  * `_wrap_pi(angle)`

    * Static helper that maps an angle to the interval `[-π, π)`.
  * `_wrap_q()`

    * Applies `_wrap_pi` to both joint angles `q[0], q[1]` in-place.
  * `_obs_dim()`

    * Returns the observation vector size implied by `obs_mode` (`minimal`, `minimal_ee`, `error_only`, `ik_error`).
  * `_forward_kinematics(q)`

    * Computes end-effector position `(x, y)` from joint angles using standard 2-link planar kinematics.
  * `_inverse_kinematics(target_xy)`

    * Computes one analytic IK solution branch (elbow-down convention) for a desired end-effector point, with numeric clipping for reachability and angle wrapping to `[-π, π)`.
  * `_distance()`

    * Returns Euclidean distance between current end-effector position and the current target.
  * `_loss()`

    * Returns squared distance error `||end_effector - target||²` (used for reward and logging).
  * `_setup_target_function_for_episode()`

    * Creates and stores `self.target_fn` for the episode based on `target_mode`:

      * string presets: `static_fixed`, `static_random`, `circle_origin`, `lissajous`
      * or a user-provided callable `f(t)->(x,y)`
  * `_update_target()`

    * Evaluates `self.target_fn` at the current simulation time `t = steps * dt` and updates `self.target`.
  * `_get_obs()`

    * Builds the observation vector according to `obs_mode`:

      * `minimal`: target `(xt,yt)` + joint angles + joint velocities
      * `minimal_ee`: same as minimal plus end-effector `(xend,yend)`
      * `error_only`: target error `(xt-xend, yt-yend)` + angles + velocities
      * `ik_error`: joint angle error relative to IK solution + velocities
  * `reset(seed=None, options=None)`

    * Resets step counter and trail, optionally reseeds RNG, initializes state either deterministically (zeros) or randomly (`random_init`), wraps angles if enabled, configures a fresh episode target function, updates target at `t=0`, returns initial `(obs, info)`, and renders if in human mode.
  * `step(action)`

    * Clips torques to limits, updates the target for the current time, computes joint accelerations from the manipulator dynamics

      * `M(q) qdd = tau - C(q,dq)dq - B dq`
    * Integrates with semi-implicit Euler (`dq ← dq + qdd dt`, `q ← q + dq dt`), wraps angles, updates trail, computes reward `-(squared_error) - λ||tau||²`, checks termination (optional `success_radius`) and truncation (`max_steps`), returns `(obs, reward, terminated, truncated, info)`, and renders if in human mode.
  * `_build_info(success, tau=None, qdd=None)`

    * Constructs the `info` dict with target, end-effector, loss, distance, success flag, and current `q/dq`; optionally includes applied torques `tau` and accelerations `qdd`.
  * `_init_pygame()`

    * Lazily initializes pygame window/surface, clock, and font; errors if pygame is unavailable.
  * `_world_to_screen(p)`

    * Converts world coordinates (meters) to screen pixel coordinates using a fixed scale centered in the window.
  * `_draw_grid(surf)`

    * Draws a light background grid and coordinate axes onto the pygame surface.
  * `render()`

    * Renders the arm, target marker, optional success radius, workspace circles, optional end-effector trail, and a HUD (step, time, distance, reward, state, modes).
    * In `human` mode, updates the display and processes quit events. In `rgb_array` mode, returns an `(H, W, 3)` numpy array of pixels.
  * `close()`

    * Shuts down pygame (if initialized) and clears rendering resources.
"""

import os
import gymnasium as gym
from IPython.display import Video, display
from gymnasium.wrappers import RecordVideo
import numpy as np

os.makedirs("videos", exist_ok=True)

env = TwoLinkReacherEnv(
    # target=(1.5, 0.0),
    target_mode="lissajous",
    render_mode="rgb_array",
    obs_mode="minimal",
    max_steps=150,
    torque_limit=2.0,
)

env = RecordVideo(
    env,
    video_folder="videos",
    episode_trigger=lambda ep: True,
    name_prefix="two_link_reacher_constant_action",
)

obs, info = env.reset(seed=42)
done = False
total_reward = 0.0

action = np.array([1.0, 1.0], dtype=np.float32)

while not done:
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

env.close()
print(f"Episode finished. Total reward = {total_reward:.3f}")
print("Video saved under ./videos/")

# Find all generated video files
video_dir = "videos"
video_files = sorted(
    [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
)

if not video_files:
    print("No video files found.")
else:
    print("Available episodes:")
    for i, f in enumerate(video_files):
        print(f"{i}: {f}")

    episode_index = -1

    video_path = os.path.join(video_dir, video_files[episode_index])
    print(f"\nDisplaying: {video_path}")
    display(Video(video_path, embed=True))

"""# Part (b): Cartesian-Space PID Control"""

def static_target_fn(t: float) -> np.ndarray:
    # Scenario 1: static target = (1.5, 0)
    return np.array([1.5, 0.0], dtype=np.float64)

def circular_target_fn(t: float) -> np.ndarray:
    # Scenario 2: circular motion around origin
    # omega = (2π/3)*0.2 = 2π/15 rad/s, r=0.5, center=(0,0)
    omega = 2.0 * np.pi / 15.0
    r = 0.5
    return np.array([r * np.cos(omega * t), r * np.sin(omega * t)], dtype=np.float64)
