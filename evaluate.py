import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import gymnasium as gym
from stable_baselines3 import PPO
from dropout_wrapper import SensorDropoutWrapper, SingleDimDropout

MODELS_DIR  = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_EVAL_EPISODES = 50

BLUE  = "#378ADD"
GREEN = "#1D9E75"
AMBER = "#BA7517"
GRAY  = "#888780"

plt.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "#FAFAF8",
    "axes.facecolor": "#FAFAF8",
})


def evaluate_model(model, env_fn, n_episodes=N_EVAL_EPISODES):
    rewards = []
    for _ in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += r
        rewards.append(total)
        env.close()
    return float(np.mean(rewards))


def smooth(values, window=5):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode="same")


print("Loading models...")
model_clean  = PPO.load(os.path.join(MODELS_DIR, "ppo_clean"))
model_robust = PPO.load(os.path.join(MODELS_DIR, "ppo_robust"))


# ── Plot 1: Learning curves ───────────────────────────────────────────────────
logs_path = os.path.join(MODELS_DIR, "training_logs.npz")
if os.path.exists(logs_path):
    logs = np.load(logs_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(logs["clean_steps"],  smooth(logs["clean_rewards"]),  color=BLUE,  lw=2, label="Baseline (no dropout)")
    ax.plot(logs["robust_steps"], smooth(logs["robust_rewards"]), color=GREEN, lw=2, ls="--", label="Robust (dropout=0.3)")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Learning curves — clean vs. robust PPO", fontsize=13, fontweight="bold", pad=12)
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.set_ylim(0, 520)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "learning_curves.png"), dpi=150)
    plt.close()
    print("Saved learning_curves.png")


# ── Plot 2: Dropout robustness sweep ─────────────────────────────────────────
print("\nEvaluating across dropout rates...")
dropout_rates   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
results_clean   = []
results_robust  = []

for d in dropout_rates:
    print(f"  dropout={d:.2f}")
    r_clean  = evaluate_model(model_clean,  lambda d=d: SensorDropoutWrapper(gym.make("CartPole-v1"), d))
    r_robust = evaluate_model(model_robust, lambda d=d: SensorDropoutWrapper(gym.make("CartPole-v1"), d))
    results_clean.append(r_clean)
    results_robust.append(r_robust)
    print(f"    baseline={r_clean:.1f}  robust={r_robust:.1f}")

np.savez(os.path.join(RESULTS_DIR, "dropout_sweep.npz"),
         dropout_rates=dropout_rates,
         results_clean=results_clean,
         results_robust=results_robust)

pct = [f"{int(d*100)}%" for d in dropout_rates]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(pct, results_clean,  color=BLUE,  marker="o", lw=2, label="Baseline")
ax.plot(pct, results_robust, color=GREEN, marker="s", lw=2, ls="--", label="Robust (trained w/ dropout)")
ax.fill_between(range(len(pct)), results_clean, results_robust, alpha=0.08, color=GREEN)
ax.set_xlabel("Sensor dropout rate at test time")
ax.set_ylabel("Mean episode reward")
ax.set_title("Robustness to sensor dropout", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 520)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "dropout_robustness.png"), dpi=150)
plt.close()
print("Saved dropout_robustness.png")


# ── Plot 3: Sensor importance ─────────────────────────────────────────────────
print("\nRunning per-sensor ablation...")
dim_names = ["Cart\nposition", "Cart\nvelocity", "Pole\nangle", "Pole ang.\nvelocity"]
dim_rewards_clean  = []
dim_rewards_robust = []

for i in range(4):
    r_clean  = evaluate_model(model_clean,  lambda i=i: SingleDimDropout(gym.make("CartPole-v1"), i))
    r_robust = evaluate_model(model_robust, lambda i=i: SingleDimDropout(gym.make("CartPole-v1"), i))
    dim_rewards_clean.append(r_clean)
    dim_rewards_robust.append(r_robust)
    print(f"  mask dim {i} ({SingleDimDropout.DIM_NAMES[i]}): baseline={r_clean:.1f}  robust={r_robust:.1f}")

x = np.arange(4)
width = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
bars1 = ax.bar(x - width/2, dim_rewards_clean,  width, color=BLUE,  label="Baseline", alpha=0.85)
bars2 = ax.bar(x + width/2, dim_rewards_robust, width, color=GREEN, label="Robust",   alpha=0.85)
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 6, f"{h:.0f}",
            ha="center", va="bottom", fontsize=9, color="#444")
ax.set_xticks(x)
ax.set_xticklabels(dim_names)
ax.set_ylabel("Mean episode reward (lower = more critical)")
ax.set_title("Sensor importance — which dimension matters most?", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 560)
ax.axhline(500, color=GRAY, lw=1, ls=":", label="Max reward")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sensor_importance.png"), dpi=150)
plt.close()
print("Saved sensor_importance.png")


# ── Plot 4: Performance gap ───────────────────────────────────────────────────
gap = [r - b for r, b in zip(results_robust, results_clean)]
colors_bar = [GREEN if g >= 0 else AMBER for g in gap]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(pct, gap, color=colors_bar, alpha=0.85, edgecolor="white", linewidth=0.5)
for bar, g in zip(bars, gap):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (4 if g >= 0 else -14),
            f"+{g:.0f}" if g >= 0 else f"{g:.0f}",
            ha="center", va="bottom", fontsize=9, color="#444")
ax.axhline(0, color=GRAY, lw=1)
ax.set_xlabel("Sensor dropout rate at test time")
ax.set_ylabel("Reward delta (robust − baseline)")
ax.set_title("Performance gap — how much does dropout training help?", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "performance_gap.png"), dpi=150)
plt.close()
print("Saved performance_gap.png")

print("\n All plots saved to results/")
print(" Run: streamlit run app.py")