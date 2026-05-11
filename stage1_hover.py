"""
STAGE 1 — Hover tại chỗ (stabilization only)
=============================================
Bài toán: spawn gần target, học GIỮ YÊN tại chỗ.
Agent không cần biết mình ở đâu — chỉ cần biết mình đang nghiêng/xoay/bay không.

Obs (9,): rpy(3) + vel(3) + ang_vel(3)
  → Tất cả đều cần về 0 để hover tốt
  → KHÔNG có pos, KHÔNG có rel_target, KHÔNG có dist
  → Đơn giản nhất có thể cho bài toán stabilization

Tại sao bỏ pos/dist?
  - Stage 1 spawn ngay dưới target (0.8 → 1.0), chỉ cần bay lên 0.2m rồi giữ
  - Cho pos vào obs chỉ làm agent bị distract: nó cố học map pos→action
    thay vì học cái cần thiết hơn là stability→action
  - Agent học vẹt: "tôi ở tọa độ X thì làm Y" thay vì "tôi đang nghiêng thì cân bằng lại"

Khi nào chuyển Stage 2?
  → steps/episode > 800 (gần MAX_STEPS) liên tiếp — drone không crash sớm
  → mean hover_ratio > 80%

NOTE về transfer sang Stage 2:
  Stage 2 dùng obs (16,) có thêm pos+rel_target+dist.
  Khi transfer: load stage1 weights, stage2 sẽ tự học thêm các chiều mới.
  Các chiều stability (rpy, vel, ang_vel) đã được học tốt → giữ nguyên.
"""

import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class Stage1HoverEnv(BaseRLAviary):

    TARGET    = np.array([0.0, 0.0, 1.0])
    SPAWN     = np.array([0.0, 0.0, 0.8])
    MAX_STEPS = 1000
    HOVER_R   = 0.15

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
        self._step_n      = 0
        self._last_action = None

    def _sample_spawn(self):
        noise = np.array([
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.10, 0.10),
        ])
        return self.SPAWN + noise

    # ── Obs (16,): Đồng nhất với Stage 2 & 3 để hỗ trợ Transfer Learning ──────

    def _observationSpace(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

    def _computeObs(self):
        s       = self._getDroneStateVector(0)
        pos = np.array([0.0, 0.0, s[2]], dtype=np.float32)
        rpy = s[7:10]
        vel = s[10:13]
        ang_vel = s[13:16]
        rel = self.TARGET - s[0:3]
        rel[0] = 0.0 # Zero-out X
        rel[1] = 0.0 # Zero-out Y
        
        dist = np.array([abs(rel[2])], dtype=np.float32)
        # Trả về 16 chiều giống Stage 2
        return np.concatenate([pos, rpy, vel, ang_vel, rel, dist]).astype(np.float32)

    # ── Step override để lưu action ───────────────────────────────────────────

    def step(self, action):
        self._last_action = np.array(action, dtype=np.float32)
        return super().step(action)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _computeReward(self):
        s    = self._getDroneStateVector(0)
        rpy  = s[7:10]
        vel  = s[10:13]
        ang_vel = s[13:16]

        # 1. Alive bonus
        r_alive = 1.0

        # 2. Hover bonus — thưởng khi gần target
        r_hover = np.exp(-np.linalg.norm(vel) * 3.0) + np.exp(-np.linalg.norm(ang_vel) * 3.0)

        # 4. Attitude penalty — luôn phạt khi nghiêng
        r_att = -np.linalg.norm(rpy[:2]) * 2.0

        # 5. Angular velocity penalty khi hover — học ra action ~0 khi đứng yên
        r_action_leash = 0.0
        if self._last_action is not None:
            # last_action[1:] là Roll, Pitch, Yaw
            r_action_leash = -np.linalg.norm(self._last_action[1:]) * 2.0
                
        self._prev_action = self._last_action.copy() if self._last_action is not None else None
        # Không có r_dist vì stage 1 không cần navigate
        return float(r_alive + r_hover + r_att + r_action_leash)

    # ── Terminated ────────────────────────────────────────────────────────────

    def _computeTerminated(self):
        s = self._getDroneStateVector(0)
        pos, rpy = s[0:3], s[7:10]
        return bool(
            pos[2] < 0.1
            or abs(rpy[0]) > 1.0  
            or abs(rpy[1]) > 1.0
        )

    def _computeTruncated(self):
        self._step_n += 1
        return self._step_n >= self.MAX_STEPS

    def _computeInfo(self):
        s    = self._getDroneStateVector(0)
        dist = float(np.linalg.norm(s[0:3] - self.TARGET))
        return {"dist_to_target": dist, "stage": 1}

    def reset(self, seed=None, options=None):
        self._step_n      = 0
        self._last_action = None
        self.INIT_XYZS    = self._sample_spawn()[np.newaxis]
        return super().reset(seed=seed, options=options)
