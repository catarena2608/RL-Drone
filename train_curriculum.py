"""
train_curriculum.py
===================
Train theo curriculum 3 stage, transfer weights giữa các stage.

Stage 1 → Stage 2 → Stage 3

Cách dùng:
    python train_curriculum.py                    # train toàn bộ từ đầu
    python train_curriculum.py --stage 2          # train từ stage 2 (cần stage1 đã xong)
    python train_curriculum.py --stage 3          # train từ stage 3 (cần stage2 đã xong)
    python train_curriculum.py --eval --stage 3   # chỉ eval stage 3

Tại sao PPO thay vì TD3?
    - PyBullet không support parallel env tốt → n_envs=1
    - Với n_envs=1, PPO và TD3 ngang nhau về sample efficiency
    - PPO đơn giản hơn, ít hyperparameter hơn, dễ debug hơn cho đồ án
    - ActionType.PID đã smooth action → không cần replay buffer của TD3

Transfer weight strategy:
    - Stage 1 → Stage 2: obs shape GIỐNG NHAU (16 → 16)
      → Load thẳng model.zip từ Stage 1 để tiếp tục train Stage 2
    - Stage 2 → Stage 3: obs shape GIỐNG NHAU (16 → 16)
      → Load thẳng model.zip, set_env, fine-tune
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from stage1_hover import Stage1HoverEnv
from stage2_nav   import Stage2NavEnv
from stage3_final import Stage3FinalEnv


# =============================================================================
# Config
# =============================================================================

STAGE_CONFIG = {
    1: {
        "env_cls":    Stage1HoverEnv,
        "timesteps":  500_000,
        "model_dir":  "models/stage1",
        "log_dir":    "logs/stage1",
        "eval_freq":  10_000,
        "n_eval_ep":  5,
    },
    2: {
        "env_cls":    Stage2NavEnv,
        "timesteps":  500_000,
        "model_dir":  "models/stage2",
        "log_dir":    "logs/stage2",
        "eval_freq":  15_000,
        "n_eval_ep":  10,
    },
    3: {
        "env_cls":    Stage3FinalEnv,
        "timesteps":  500_000,
        "model_dir":  "models/stage3",
        "log_dir":    "logs/stage3",
        "eval_freq":  20_000,
        "n_eval_ep":  10,
    },
}

PPO_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-1.0,
    ),
    verbose=1,
    seed=42,
)


# =============================================================================
# Helpers
# =============================================================================

def make_env_fn(env_cls, gui=False):
    def _init():
        return Monitor(env_cls(gui=gui))
    return _init


def make_callbacks(cfg, eval_env):
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg["model_dir"], "best"),
        log_path=cfg["log_dir"],
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=cfg["n_eval_ep"],
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=cfg["eval_freq"] * 2,
        save_path=os.path.join(cfg["model_dir"], "checkpoints"),
        name_prefix="ppo",
        verbose=0,
    )
    return CallbackList([eval_cb, ckpt_cb])


# =============================================================================
# Train stage 1
# =============================================================================

def train_stage1(args):
    cfg = STAGE_CONFIG[1]
    os.makedirs(cfg["model_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"],   exist_ok=True)

    train_env = make_vec_env(make_env_fn(cfg["env_cls"]), n_envs=1, seed=42)
    eval_env  = make_vec_env(make_env_fn(cfg["env_cls"]), n_envs=1, seed=99)

    model = PPO(
        env=train_env,
        tensorboard_log=cfg["log_dir"],
        **PPO_KWARGS,
    )

    print(f"\n{'='*60}\n  STAGE 1: Hover cố định\n{'='*60}")
    model.learn(
        total_timesteps=cfg["timesteps"],
        callback=make_callbacks(cfg, eval_env),
        progress_bar=True,
    )

    save_path = os.path.join(cfg["model_dir"], "stage1_final")
    model.save(save_path)
    print(f"\n✅ Stage 1 saved: {save_path}.zip")

    train_env.close()
    eval_env.close()
    return model


# =============================================================================
# Train stage 2 — transfer từ stage 1
# Obs shape GIỐNG NHAU (16 = 16) → load thẳng model.zip
# =============================================================================

def train_stage2(args):
    cfg     = STAGE_CONFIG[2]
    cfg_s1  = STAGE_CONFIG[1]
    os.makedirs(cfg["model_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"],   exist_ok=True)

    train_env = make_vec_env(make_env_fn(cfg["env_cls"]), n_envs=1, seed=42)
    eval_env  = make_vec_env(make_env_fn(cfg["env_cls"]), n_envs=1, seed=99)

    # Load Stage 1 model để transfer
    best_path  = os.path.join(cfg_s1["model_dir"], "best", "best_model.zip")
    final_path = os.path.join(cfg_s1["model_dir"], "stage1_final.zip")
    src = best_path if os.path.exists(best_path) else (final_path if os.path.exists(final_path) else None)

    if src:
        print(f"\n📦 Load Stage1 weights từ: {src}")
        model = PPO.load(src, env=train_env, tensorboard_log=cfg["log_dir"])
        # Fine-tune với learning_rate nhỏ
        model.learning_rate = 1e-4
    else:
        print("\n⚠️ Không tìm thấy Stage1 model, train từ đầu Stage 2")
        model = PPO(
            env=train_env,
            tensorboard_log=cfg["log_dir"],
            **{**PPO_KWARGS, "learning_rate": 1e-4},
        )

    print(f"\n{'='*60}\n  STAGE 2: Navigate từ gần (spawn < 1m)\n{'='*60}")
    model.learn(
        total_timesteps=cfg["timesteps"],
        callback=make_callbacks(cfg, eval_env),
        progress_bar=True,
    )

    save_path = os.path.join(cfg["model_dir"], "stage2_final")
    model.save(save_path)
    print(f"\n✅ Stage 2 saved: {save_path}.zip")

    train_env.close()
    eval_env.close()
    return model


# =============================================================================
# Train stage 3 — transfer TRỰC TIẾP từ stage 2
# Obs shape GIỐNG NHAU (16 = 16) → load thẳng model.zip
# =============================================================================

def train_stage3(args):
    cfg    = STAGE_CONFIG[3]
    cfg_s2 = STAGE_CONFIG[2]
    os.makedirs(cfg["model_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"],   exist_ok=True)

    train_env = make_vec_env(make_env_fn(cfg["env_cls"]), n_envs=1, seed=42)
    eval_env  = make_vec_env(make_env_fn(cfg["env_cls"]), n_envs=1, seed=99)

    # Tìm best model của Stage 2
    best_path  = os.path.join(cfg_s2["model_dir"], "best", "best_model.zip")
    final_path = os.path.join(cfg_s2["model_dir"], "stage2_final.zip")

    if os.path.exists(best_path):
        src = best_path
    elif os.path.exists(final_path):
        src = final_path
    else:
        raise FileNotFoundError(
            "Không tìm thấy Stage2 model. Hãy train Stage 2 trước:\n"
            "  python train_curriculum.py --stage 2"
        )

    print(f"\n📦 Load Stage2 weights từ: {src}")
    # Load với env mới — SB3 tự điều chỉnh
    # Cần set tensorboard_log ở đây để stage 3 log tiếp vào logs/stage3
    model = PPO.load(src, env=train_env, tensorboard_log=cfg["log_dir"])

    # Fine-tune với learning_rate nhỏ để giữ knowledge từ Stage 2
    model.learning_rate = 5e-5

    print(f"\n{'='*60}\n  STAGE 3: Navigate + Hover lâu nhất (spawn < 3m)\n{'='*60}")
    model.learn(
        total_timesteps=cfg["timesteps"],
        callback=make_callbacks(cfg, eval_env),
        progress_bar=True,
        reset_num_timesteps=False,  # giữ tensorboard liên tục
    )

    save_path = os.path.join(cfg["model_dir"], "stage3_final")
    model.save(save_path)
    print(f"\n✅ Stage 3 saved: {save_path}.zip")

    train_env.close()
    eval_env.close()
    return model


# =============================================================================
# Evaluate
# =============================================================================

def evaluate(stage, gui=False, n_episodes=10):
    cfg = STAGE_CONFIG[stage]
    env_cls = cfg["env_cls"]

    best_path  = os.path.join(cfg["model_dir"], "best", "best_model.zip")
    final_path = os.path.join(cfg["model_dir"], f"stage{stage}_final.zip")
    src = best_path if os.path.exists(best_path) else final_path

    print(f"\nEval Stage {stage} từ: {src}")
    model = PPO.load(src)
    env   = Monitor(env_cls(gui=gui))

    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r, steps, hover_steps = 0, 0, 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += r
            steps   += 1
            if info.get("dist_to_target", 999) < 0.15:
                hover_steps += 1

        results.append({
            "reward": total_r,
            "steps":  steps,
            "hover_steps": hover_steps,
            "hover_ratio": hover_steps / max(steps, 1),
        })
        print(f"  Ep {ep+1:2d}: reward={total_r:7.1f} | steps={steps:4d} | "
              f"hover={hover_steps:4d} ({hover_steps/max(steps,1)*100:.0f}%)")

    rews  = [r["reward"]      for r in results]
    hovrs = [r["hover_ratio"] for r in results]
    print(f"\n{'='*50}")
    print(f"  Mean reward:      {np.mean(rews):.1f} ± {np.std(rews):.1f}")
    print(f"  Mean hover ratio: {np.mean(hovrs)*100:.1f}%")
    print(f"{'='*50}\n")
    env.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",    type=int,  default=0,     help="0=all, 1/2/3=specific stage")
    parser.add_argument("--eval",     action="store_true",      help="Chỉ eval, không train")
    parser.add_argument("--gui",      action="store_true")
    parser.add_argument("--episodes", type=int,  default=10)
    args = parser.parse_args()

    if args.eval:
        stage = args.stage if args.stage > 0 else 3
        evaluate(stage, gui=args.gui, n_episodes=args.episodes)
    else:
        if args.stage == 0:
            # Train toàn bộ curriculum
            train_stage1(args)
            train_stage2(args)
            train_stage3(args)
        elif args.stage == 1:
            train_stage1(args)
        elif args.stage == 2:
            train_stage2(args)
        elif args.stage == 3:
            train_stage3(args)

        print("\n--- Eval sau khi train ---")
        evaluate(args.stage if args.stage > 0 else 3, n_episodes=args.episodes)
