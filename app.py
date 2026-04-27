import os
import numpy as np
import gymnasium as gym
import streamlit as st
import matplotlib.pyplot as plt
from dropout_wrapper import SensorDropoutWrapper, SingleDimDropout

MODELS_DIR  = "models"
RESULTS_DIR = "results"

st.set_page_config(page_title="CartPole Sensor Dropout", page_icon="🎯", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0E0E10; color: #E8E8E8; }
.main { background: #0E0E10; }
.block-container { max-width: 760px; padding-top: 2rem; }
.hero { background: #151518; border: 1px solid #2A2A30; border-radius: 12px; padding: 28px 32px; margin-bottom: 24px; }
.hero h1 { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 600; color: #F0F0F0; margin: 0 0 6px; }
.hero p  { font-size: 13px; color: #888; margin: 0; line-height: 1.6; }
.metric-row { display: flex; gap: 12px; margin-bottom: 20px; }
.metric { flex: 1; background: #151518; border: 1px solid #2A2A30; border-radius: 10px; padding: 16px 18px; }
.metric .label { font-size: 11px; color: #666; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px; }
.metric .value { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 600; color: #F0F0F0; }
.metric .sub   { font-size: 11px; color: #555; margin-top: 2px; }
.tag { display: inline-block; font-size: 11px; padding: 3px 10px; border-radius: 20px; margin-right: 6px; font-family: 'JetBrains Mono', monospace; }
.tag-blue  { background: #0D2A4A; color: #6BB8F5; border: 1px solid #1A4A7A; }
.tag-green { background: #0B2E22; color: #4EC9A0; border: 1px solid #175C3F; }
.tag-amber { background: #2E2008; color: #E8A740; border: 1px solid #5C3C10; }
hr.dim { border: none; border-top: 1px solid #2A2A30; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎯 CartPole — Sensor Dropout Robustness</h1>
  <p>Comparing a baseline PPO agent vs. a robust agent trained under 30% observation noise.<br>
  Adjust the slider to simulate sensor failure and watch performance diverge.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    from stable_baselines3 import PPO
    try:
        m_clean  = PPO.load(os.path.join(MODELS_DIR, "ppo_clean"))
        m_robust = PPO.load(os.path.join(MODELS_DIR, "ppo_robust"))
        return m_clean, m_robust
    except Exception as e:
        return None, None

model_clean, model_robust = load_models()
models_loaded = model_clean is not None

if not models_loaded:
    st.warning("Models not found. Run `python train.py` first.")


# ── Live evaluation ───────────────────────────────────────────────────────────
st.markdown("### Live evaluation")
col1, col2 = st.columns([2, 1])
with col1:
    dropout = st.slider("Sensor dropout rate", 0.0, 0.75, 0.0, 0.05)
with col2:
    n_eps = st.selectbox("Episodes", [10, 25, 50], index=1)

if st.button("▶  Run evaluation", disabled=not models_loaded):
    def run_eval(model, dropout_rate, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            env = SensorDropoutWrapper(gym.make("CartPole-v1"), dropout_rate)
            obs, _ = env.reset()
            total, done = 0.0, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += r
            rewards.append(total)
            env.close()
        return rewards

    with st.spinner("Running episodes..."):
        r_clean  = run_eval(model_clean,  dropout, n_eps)
        r_robust = run_eval(model_robust, dropout, n_eps)

    mean_c = np.mean(r_clean)
    mean_r = np.mean(r_robust)
    gap    = mean_r - mean_c

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric">
        <div class="label">Baseline agent</div>
        <div class="value">{mean_c:.0f}</div>
        <div class="sub">mean reward / 500 max</div>
      </div>
      <div class="metric">
        <div class="label">Robust agent</div>
        <div class="value">{mean_r:.0f}</div>
        <div class="sub">mean reward / 500 max</div>
      </div>
      <div class="metric">
        <div class="label">Gap (robust − baseline)</div>
        <div class="value" style="color:{'#4EC9A0' if gap>=0 else '#E8A740'}">{"+" if gap>=0 else ""}{gap:.0f}</div>
        <div class="sub">{'Robust wins' if gap>=0 else 'Baseline wins'} at dropout={dropout:.2f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#151518")
    ax.set_facecolor("#151518")
    ax.hist(r_clean,  bins=15, color="#378ADD", alpha=0.7, label="Baseline", edgecolor="none")
    ax.hist(r_robust, bins=15, color="#1D9E75", alpha=0.7, label="Robust",   edgecolor="none")
    ax.axvline(mean_c, color="#378ADD", lw=1.5, ls="--")
    ax.axvline(mean_r, color="#1D9E75", lw=1.5, ls="--")
    ax.set_xlabel("Episode reward", color="#888", fontsize=11)
    ax.set_ylabel("Count", color="#888", fontsize=11)
    ax.tick_params(colors="#666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2A30")
    ax.legend(facecolor="#1A1A1E", edgecolor="#2A2A30", labelcolor="#CCC", fontsize=10)
    st.pyplot(fig)
    plt.close()


# ── Pre-generated plots ───────────────────────────────────────────────────────
st.markdown('<hr class="dim">', unsafe_allow_html=True)
st.markdown("### Experiment results")

plots = [
    ("learning_curves.png",    "Learning curves",             "Reward over training timesteps"),
    ("dropout_robustness.png", "Dropout robustness sweep",    "Performance across dropout rates 0–75%"),
    ("sensor_importance.png",  "Sensor importance (ablation)","Which observation dimension is most critical?"),
    ("performance_gap.png",    "Performance gap",             "Reward advantage of robust agent per dropout level"),
]

any_plot = False
for fname, title, caption in plots:
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        any_plot = True
        st.markdown(f"**{title}** — *{caption}*")
        st.image(path, use_container_width=True)
        st.markdown('<hr class="dim">', unsafe_allow_html=True)

if not any_plot:
    st.info("Run `python evaluate.py` to generate plots — they'll appear here automatically.")


# ── Key findings ──────────────────────────────────────────────────────────────
st.markdown("### Key findings")
st.markdown("""
<span class="tag tag-green">Safety insight</span> Training with sensor dropout acts as observation regularisation — the robust agent learns policies that don't over-rely on any single state dimension.<br><br>
<span class="tag tag-blue">Most critical sensor</span> Masking <b>pole angular velocity</b> alone causes the largest performance collapse.<br><br>
<span class="tag tag-amber">Tradeoff</span> Robust agent trains slightly slower but generalises significantly better under distribution shift.
""", unsafe_allow_html=True)