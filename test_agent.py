"""
test_agent.py
=============
Test và visualize agent đã train ở bất kỳ stage nào.

Cách dùng:
    python test_agent.py --stage 1                        # test stage 1, tự tìm best model
    python test_agent.py --stage 2 --gui                  # test stage 2 với PyBullet GUI
    python test_agent.py --stage 3 --episodes 20          # test stage 3, 20 episodes
    python test_agent.py --model path/to/model.zip --stage 3  # chỉ định model cụ thể
    python test_agent.py --stage 3 --gui --record         # record video
    python test_agent.py --stage 3 --no-deterministic     # test stochastic policy
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from stage1_hover import Stage1HoverEnv
from stage2_nav   import Stage2NavEnv
from stage3_final import Stage3FinalEnv


# =============================================================================
# Config
# =============================================================================

STAGE_ENV = {
    1: Stage1HoverEnv,
    2: Stage2NavEnv,
    3: Stage3FinalEnv,
}

STAGE_MODEL_PATHS = {
    1: ["models/stage1/best/best_model.zip", "models/stage1/stage1_final.zip"],
    2: ["models/stage2/best/best_model.zip", "models/stage2/stage2_final.zip"],
    3: ["models/stage3/best/best_model.zip", "models/stage3/stage3_final.zip"],
}


# =============================================================================
# Helpers
# =============================================================================

def find_model(stage, override_path=None):
    """Tìm model theo thứ tự ưu tiên: override → best → final."""
    if override_path:
        if not override_path.endswith(".zip"):
            override_path += ".zip"
        if os.path.exists(override_path):
            return override_path
        raise FileNotFoundError(f"Không tìm thấy model: {override_path}")

    for path in STAGE_MODEL_PATHS[stage]:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Không tìm thấy model cho Stage {stage}.\n"
        f"Đã tìm ở:\n" + "\n".join(f"  - {p}" for p in STAGE_MODEL_PATHS[stage]) +
        f"\nHãy train trước:\n  python train_curriculum.py --stage {stage}"
    )


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_episode(ep, res):
    # Thay đổi log để phản ánh đúng Stage 1 mới
    print(f"  Ep {ep:3d}: reward={res['total_reward']:8.2f} | steps={res['steps']:4d} "
          f"| avg_tilt={np.degrees(res['avg_tilt']):.1f}° | avg_vel={res['avg_vel']:.2f}m/s "
          f"| stable={res['stable_ratio']*100:.0f}%")   


# =============================================================================
# Single episode runner
# =============================================================================

def run_episode(model, env, deterministic=True, render_delay=0.0):
    obs, _ = env.reset()
    done   = False

    total_reward = 0.0
    steps        = 0
    
    # Các thông số đánh giá độ "Tĩnh" (đặc thù Stage 1 mới)
    stable_steps = 0 
    all_rpy      = []
    all_vel      = []
    trajectory   = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps        += 1
        
        # State từ KIN obs: rpy=obs[3:6], vel=obs[6:9]
        rpy = obs[3:6]
        vel = obs[6:9]
        
        all_rpy.append(rpy)
        all_vel.append(vel)
        trajectory.append(obs[0:3].copy())

        # Định nghĩa "Stable" cho Stage 1: Nghiêng ít (< 5 độ) và di chuyển chậm
        if np.linalg.norm(rpy[:2]) < 0.087 and np.linalg.norm(vel) < 0.1:
            stable_steps += 1

        if render_delay > 0:
            time.sleep(render_delay)

    return {
        "total_reward": total_reward,
        "steps":        steps,
        "stable_steps": stable_steps,
        "stable_ratio": stable_steps / max(steps, 1),
        "avg_tilt":     np.mean([np.linalg.norm(r[:2]) for r in all_rpy]),
        "avg_vel":      np.mean([np.linalg.norm(v) for v in all_vel]),
        "trajectory":   np.array(trajectory),
        "info":         info
    }

# =============================================================================
# Main test function
# =============================================================================

def plot_trajectory(trajectory, stage, episode, save_dir):
    fig = plt.figure(figsize=(12, 5))
    
    # Subplot 1: 3D Path
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ax1.set_title("3D Trajectory (Drifting is OK in Stage 1)")
    
    # Subplot 2: Altitude (Z) over time - Quan trọng nhất Stage 1
    ax2 = fig.add_subplot(122)
    ax2.plot(trajectory[:, 2], color='blue', label='Current Z')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Target Z')
    ax2.set_ylim(0, 1.5)
    ax2.set_title("Altitude Stability")
    ax2.legend()

    plt.savefig(os.path.join(save_dir, f"stage{stage}_ep{episode}.png"))
    plt.close()

def test(args):
    # ── Tạo thư mục lưu kết quả ──────────────────────────────────────────────
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    model_path = find_model(args.stage, args.model)
    print_header(f"Test Stage {args.stage}")
    print(f"  Model    : {model_path}")
    print(f"  Episodes : {args.episodes}")
    print(f"  GUI      : {args.gui}")
    print(f"  Policy   : {'deterministic' if args.deterministic else 'stochastic'}")

    model = PPO.load(model_path)

    # ── Tạo env ────────────────────────────────────────────────────────────────
    env_cls = STAGE_ENV[args.stage]
    env     = Monitor(env_cls(gui=args.gui))

    # Nếu GUI: chạy chậm hơn để quan sát
    render_delay = 1.0 / 30.0 if args.gui else 0.0

    # ── Chạy episodes ──────────────────────────────────────────────────────────
    all_results = []

    for ep in range(1, args.episodes + 1):
        result = run_episode(
            model, env,
            deterministic=args.deterministic,
            render_delay=render_delay,
        )
        all_results.append(result)
        print_episode(ep, result)
        
        # Save plot nếu yêu cầu
        if args.save_dir:
            plot_trajectory(result["trajectory"], args.stage, ep, args.save_dir)

    env.close()
    
    if args.save_dir:
        print(f"\n📊 Đã lưu các biểu đồ quỹ đạo vào: {args.save_dir}")

    # ── Tổng kết ───────────────────────────────────────────────────────────────
    print_summary(all_results, args.stage)


# =============================================================================
# Summary statistics
# =============================================================================

def print_summary(results, stage):
    rewards = [r["total_reward"] for r in results]
    tilts   = [np.degrees(r["avg_tilt"]) for r in results]
    vels    = [r["avg_vel"] for r in results]
    stables = [r["stable_ratio"] for r in results]

    print(f"\n{'='*60}")
    print(f"  TỔNG KẾT — Stage {stage} ({len(results)} episodes)")
    print(f"{'='*60}")
    print(f"  Reward trung bình : {np.mean(rewards):.2f}")
    print(f"  Độ nghiêng TB     : {np.mean(tilts):.2f} °")
    print(f"  Vận tốc TB        : {np.mean(vels):.2f} m/s")
    print(f"  Tỉ lệ ổn định     : {np.mean(stables)*100:.1f} %")

    if stage == 1:
        if np.mean(stables) > 0.7 and np.mean(tilts) < 5.0:
            print("  ✅ Tuyệt vời! Drone đã học được cách 'đứng hình'. Sẵn sàng cho Stage 2.")
        else:
            print("  ❌ Drone vẫn còn lắc hoặc trôi quá nhanh. Cần chỉnh lại Reward/Obs.")

    elif stage == 2:
        if np.mean(stables) > 0.7 and np.mean(rewards) > 100:
            print("  ✅ Agent navigate tốt — sẵn sàng chuyển Stage 3")
        elif np.mean(stables) > 0.3:
            print("  ⚠️  Agent đến được đích nhưng không ổn định — train thêm")
        else:
            print("  ❌ Agent không navigate được — kiểm tra obs rel_target")

    elif stage == 3:
        hr = np.mean(hover_ratios   = [r["stable_ratio"] for r in results])
        if hr > 0.5:
            print(f"  ✅ Agent hover {hr*100:.0f}% thời gian — kết quả tốt!")
        elif hr > 0.2:
            print(f"  ⚠️  Agent hover {hr*100:.0f}% — cần train thêm")
        else:
            print(f"  ❌ Agent hover {hr*100:.0f}% — xem lại reward stage 3")

    print(f"  {'─'*40}\n")


# =============================================================================
# Quick sanity check — test env không cần model
# =============================================================================

def sanity_check(stage):
    """
    Chạy env với random action để kiểm tra obs/reward không bị NaN/Inf.
    Dùng trước khi train để xác nhận env đúng.
    """
    print_header(f"Sanity Check — Stage {stage} env (random actions)")
    env_cls = STAGE_ENV[stage]
    env     = env_cls(gui=False)

    obs, _ = env.reset()
    print(f"  Obs shape  : {obs.shape}")
    print(f"  Obs sample : {np.round(obs, 3)}")
    print(f"  Action space: {env.action_space}")
    print(f"  Obs space   : {env.observation_space}")

    total_r = 0
    nan_count = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r

        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            nan_count += 1
            print(f"  ⚠️  NaN/Inf trong obs tại step {step}!")

        if np.isnan(r) or np.isinf(r):
            print(f"  ⚠️  NaN/Inf trong reward tại step {step}: {r}")

        if terminated or truncated:
            print(f"  Episode kết thúc tại step {step+1}")
            break

    env.close()

    print(f"\n  ✅ Sanity check xong")
    print(f"  Obs NaN/Inf : {nan_count} lần")
    print(f"  Total reward: {total_r:.2f}")
    print(f"  Info mẫu    : {info}\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test drone agent")

    parser.add_argument("--stage",          type=int,  default=3,
                        help="Stage để test: 1, 2, hoặc 3")
    parser.add_argument("--model",          type=str,  default=None,
                        help="Đường dẫn model .zip (tuỳ chọn, tự tìm nếu không có)")
    parser.add_argument("--episodes",       type=int,  default=10)
    parser.add_argument("--gui",            action="store_true",
                        help="Bật PyBullet GUI")
    parser.add_argument("--no-deterministic", action="store_true",
                        help="Dùng stochastic policy thay vì deterministic")
    parser.add_argument("--sanity",         action="store_true",
                        help="Chỉ chạy sanity check env (không cần model)")
    parser.add_argument("--save-dir",       type=str,  default="test_output",
                        help="Thư mục lưu kết quả test (biểu đồ trajectory)")

    args = parser.parse_args()
    args.deterministic = not args.no_deterministic

    if args.stage not in STAGE_ENV:
        print(f"❌ Stage phải là 1, 2, hoặc 3. Nhận được: {args.stage}")
        sys.exit(1)

    if args.sanity:
        sanity_check(args.stage)
    else:
        test(args)
