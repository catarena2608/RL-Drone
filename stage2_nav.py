"""
STAGE 2 — Navigate từ gần (random spawn trong 1m)
==================================================
Thêm một yếu tố so với Stage 1: spawn NGẪU NHIÊN trong vùng 1m quanh target.
Target vẫn cố định tại (0, 0, 1.0).

Obs thêm rel_target (3,) + dist (1,) = 16 chiều
  → Agent cần biết mình đang ở đâu so với target

Transfer từ Stage 1:
  model = PPO.load("stage1_best")
  model.set_env(Stage2NavEnv())
  model.learn(100_000)   # fine-tune, không train từ đầu

Khi nào chuyển Stage 3?
  → mean_reward > 150 / episode
  → success_rate (dist < 0.15 cuối episode) > 70%
"""

import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class Stage2NavEnv(BaseRLAviary):
    """Stage 2: random spawn trong vòng 1m, target cố định."""

    TARGET    = np.array([0.0, 0.0, 1.0])
    MAX_STEPS = 800
    HOVER_R   = 0.15
    SPAWN_R   = 1.0    # spawn trong vòng tròn 1m

    def __init__(self, gui=False):
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            physics=Physics.PYB,
            gui=gui,
            obs=ObservationType.KIN,
            act=ActionType.PID,
            initial_xyzs=self._sample_spawn()[np.newaxis],
        )
        self._step_n    = 0
        self._hover_n   = 0   # đếm steps trong vùng hover

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _sample_spawn(self):
        """Random trong vòng SPAWN_R quanh target, tránh spawn quá gần."""
        while True:
            offset = np.array([
                np.random.uniform(-self.SPAWN_R, self.SPAWN_R),
                np.random.uniform(-self.SPAWN_R, self.SPAWN_R),
                np.random.uniform(-0.5, 0.5),
            ])
            pos = self.TARGET + offset
            pos[2] = max(pos[2], 0.2)   # không spawn dưới đất
            if 0.3 < np.linalg.norm(offset) <= self.SPAWN_R:
                return pos

    # ── Obs: KIN (12) + rel_target (3) + dist (1) = 16 ────────────────────────
    def _observationSpace(self):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

    def _computeObs(self):
        s       = self._getDroneStateVector(0)   # (20,) — luôn đúng mọi version
        pos     = s[0:3]
        rpy     = s[7:10]
        vel     = s[10:13]
        ang_vel = s[13:16]
        rel     = self.TARGET - pos
        dist    = float(np.linalg.norm(rel))
        return np.concatenate([pos, rpy, vel, ang_vel, rel, [dist]]).astype(np.float32)

    # ── Reward ─────────────────────────────────────────────────────────────────
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos   = state[0:3]
        vel   = state[10:13]
        rpy   = state[7:10]
        ang_vel = state[13:16]
        dist  = np.linalg.norm(pos - self.TARGET)
        in_hover = dist < self.HOVER_R
        if not in_hover:
            # GIAI ĐOẠN 1: Điều hướng (Navigation)
            # Phạt khoảng cách nhưng không quá nặng để tránh nó cuống cuồng bay nhanh
            r_nav = -dist * 1.0 
            # Thưởng vận tốc hướng về tâm
            reward = r_nav
        else:

            
            # GIAI ĐOẠN 2: Giữ yên (Stabilization) - Thừa hưởng từ Stage 1
            # Khi đã vào vùng target, dist không còn quan trọng, quan trọng là "Tĩnh"
            r_stay = 2.0 # Thưởng vì đã đứng đúng vùng
            r_static = np.exp(-np.linalg.norm(vel) * 3.0) # Thưởng cực lớn nếu vận tốc = 0
            r_stable = -np.linalg.norm(ang_vel) * 0.5     # Phạt nếu còn rung lắc
            reward = r_stay + r_static + r_stable

        r_att = -np.linalg.norm(rpy[:2]) * 1.0

        return float(reward + r_att)

    # ── Terminated ─────────────────────────────────────────────────────────────
    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos, rpy = state[0:3], state[7:10]
        return bool(
            pos[2] < 0.05
            or abs(rpy[0]) > 1.0 or abs(rpy[1]) > 1.0
            or np.linalg.norm(pos - self.TARGET) > 3.0
        )

    def _computeTruncated(self):
        self._step_n += 1
        return self._step_n >= self.MAX_STEPS

    def _computeInfo(self):
        dist = float(np.linalg.norm(self._getDroneStateVector(0)[0:3] - self.TARGET))
        return {
            "dist_to_target": dist,
            "success": dist < self.HOVER_R,
            "stage": 2,
        }

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        self._step_n  = 0
        self._hover_n = 0
        self.INIT_XYZS = self._sample_spawn()[np.newaxis]
        return super().reset(seed=seed, options=options)
