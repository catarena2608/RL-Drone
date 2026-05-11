"""
STAGE 3 — Bài toán đầy đủ (random spawn 3m, giữ lâu nhất có thể)
==================================================================
Đây là bài toán cuối cùng:
  - Spawn NGẪU NHIÊN trong vòng 3m quanh target
  - Target CỐ ĐỊNH tại (0, 0, 1.0)
  - Không có done khi đến đích — episode chỉ kết thúc khi crash / timeout
  - Agent được thưởng mỗi step khi giữ được trong vùng hover
  → "Giữ lâu nhất có thể" = tối đa hoá hover_steps

Transfer từ Stage 2:
  model = PPO.load("stage2_best")
  model.set_env(Stage3FinalEnv())
  model.learn(300_000)

Obs (16,): giống Stage 2 — QUAN TRỌNG: phải giữ nguyên obs shape
           để transfer weights không bị lỗi dimension mismatch

Reward:
  - Giảm weight r_dist để không quá aggressively rush đến target
  - Tăng r_hover vì mục tiêu chính là TIME AT TARGET
  - Thêm r_alive: +0.1 mỗi step chỉ cần còn sống
    → khuyến khích bay bền vững, không liều lĩnh
"""

import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class Stage3FinalEnv(BaseRLAviary):
    """
    Bài toán hoàn chỉnh: navigate + hover lâu nhất có thể.
    Obs shape (16,) giống Stage 2 để transfer weights hoạt động.
    """

    TARGET    = np.array([0.0, 0.0, 1.0])
    MAX_STEPS = 2000      # ~8s — đủ dài để thấy sự khác biệt giữa agents
    HOVER_R   = 0.15      # m
    SPAWN_R   = 3.0       # m — spawn xa hơn nhiều so với Stage 2

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
        self._step_n  = 0
        self._hover_n = 0   # metric quan trọng nhất: đếm hover steps

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _sample_spawn(self):
        """Random trong vòng SPAWN_R, cách target ít nhất 0.5m."""
        while True:
            offset = np.array([
                np.random.uniform(-self.SPAWN_R, self.SPAWN_R),
                np.random.uniform(-self.SPAWN_R, self.SPAWN_R),
                np.random.uniform(-1.0, 1.5),
            ])
            pos = self.TARGET + offset
            pos[2] = max(pos[2], 0.2)
            if 0.5 < np.linalg.norm(offset) <= self.SPAWN_R:
                return pos

    # ── Obs (16,) — giữ nguyên shape so với Stage 2 ───────────────────────────
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
        dist  = np.linalg.norm(pos - self.TARGET)

        in_hover = dist < self.HOVER_R

        # 1. Alive bonus — mỗi step còn sống đều có reward nhỏ
        #    → agent học cách bay an toàn, không liều
        r_alive = 0.1

        # 2. Distance penalty — nhẹ hơn Stage 2 vì hover mới là ưu tiên
        r_dist  = -dist * 1.0

        # 3. Hover bonus — tăng lên so với Stage 2, đây là mục tiêu chính
        #    Dense: cộng mỗi step → agent muốn ở đây CÀNG LÂU CÀNG TỐT
        r_hover = 5.0 if in_hover else 0.0

        # 4. Smooth penalty
        r_vel = -np.linalg.norm(vel) * 0.05

        # 5. Attitude penalty
        r_att = -(abs(rpy[0]) + abs(rpy[1])) * 0.1

        # Update hover counter (dùng trong info)
        if in_hover:
            self._hover_n += 1

        return float(r_alive + r_dist + r_hover + r_vel + r_att)

    # ── Terminated — CHỈ crash/flip, KHÔNG terminate khi đến đích ────────────
    def _computeTerminated(self):
        """
        Quan trọng: KHÔNG terminate khi đến target.
        Agent phải học cách GIỮ vị trí, không phải "chạm rồi xong".
        Kết thúc sớm chỉ khi thực sự thất bại.
        """
        state = self._getDroneStateVector(0)
        pos, rpy = state[0:3], state[7:10]
        return bool(
            pos[2] < 0.05
            or abs(rpy[0]) > 1.0 or abs(rpy[1]) > 1.0
            or np.linalg.norm(pos - self.TARGET) > 5.0  # xa hơn Stage 2
        )

    def _computeTruncated(self):
        self._step_n += 1
        return self._step_n >= self.MAX_STEPS

    def _computeInfo(self):
        dist = float(np.linalg.norm(self._getDroneStateVector(0)[0:3] - self.TARGET))
        return {
            "dist_to_target": dist,
            "hover_steps":    self._hover_n,
            "hover_ratio":    self._hover_n / max(self._step_n, 1),
            "stage": 3,
        }

    def reset(self, seed=None, options=None):
        self._step_n  = 0
        self._hover_n = 0
        self.INIT_XYZS = self._sample_spawn()[np.newaxis]
        return super().reset(seed=seed, options=options)
