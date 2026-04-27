# CartPole — Sensor Dropout Robustness Study

An empirical RL experiment comparing a **baseline PPO agent** vs. a **robust PPO agent** trained under 30% observation noise on OpenAI Gymnasium's CartPole-v1. The project investigates how sensor dropout during training acts as observation regularisation and improves generalisation under distribution shift.

---

## Research Question

> Does training a PPO agent with randomised sensor dropout produce a more robust policy under observation noise at test time — and which state dimensions matter most?

---

## Key Findings

- **Safety insight:** Training with sensor dropout acts as observation regularisation — the robust agent learns policies that don't over-rely on any single state dimension.
- **Most critical sensor:** Masking **pole angular velocity** alone causes the largest performance collapse in both agents.
- **Tradeoff:** The robust agent trains slightly slower but generalises significantly better under distribution shift.

---

## Architecture

```
CartPole-v1 (Gymnasium)
        ↓
SensorDropoutWrapper        ← randomly zeros observation dims at each step
        ↓
PPO Agent (Stable Baselines3)
        ↓
┌──────────────────────────────┐
│  Baseline: trained clean     │  dropout_rate = 0.0
│  Robust:   trained noisy     │  dropout_rate = 0.3
└──────────────────────────────┘
        ↓
Evaluation sweep: dropout 0.0 → 0.75
        ↓
Streamlit dashboard (live eval + result plots)
```

---

## Project Structure

```
cartpole-rl-dropout/
├── train.py               # Trains baseline and robust PPO agents
├── evaluate.py            # Sweeps dropout rates, generates result plots
├── dropout_wrapper.py     # SensorDropoutWrapper + SingleDimDropout (ablation)
├── app.py                 # Streamlit dashboard — live evaluation & plots
├── models/
│   ├── ppo_clean          # Saved baseline agent
│   └── ppo_robust         # Saved robust agent
├── results/
│   ├── learning_curves.png
│   ├── dropout_robustness.png
│   ├── sensor_importance.png
│   └── performance_gap.png
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/JananiVenk/Cartpole-RL-Effect-of-dropout.git
cd Cartpole-RL-Effect-of-dropout
pip install -r requirements.txt
```

---

## Usage

**1. Train both agents**
```bash
python train.py
```
Trains a baseline PPO agent (no dropout) and a robust PPO agent (30% sensor dropout). Saves both to `models/`.

**2. Evaluate and generate plots**
```bash
python evaluate.py
```
Sweeps dropout rates from 0.0 to 0.75, runs sensor ablation (masking each dimension independently), and saves plots to `results/`.

**3. Launch the dashboard**
```bash
streamlit run app.py
```
Opens an interactive UI where you can:
- Adjust the sensor dropout rate with a slider (0–75%)
- Run live evaluation episodes for both agents
- Compare mean reward and see reward distributions
- Browse all experiment result plots

---

## Experiment Design

### Agents

| Agent | Training Dropout | Description |
|-------|-----------------|-------------|
| Baseline | 0.0 | Standard PPO on clean observations |
| Robust | 0.3 | PPO trained with 30% sensor dropout per step |

### Evaluation

- Dropout rates swept: 0.0, 0.05, 0.10, ..., 0.75
- Episodes per configuration: 25 (configurable in dashboard)
- Max reward per episode: 500 (CartPole-v1)

### Sensor Ablation

Each of CartPole's 4 observation dimensions is masked independently to measure its contribution to policy performance:

| Dim | Observation | Impact when masked |
|-----|-------------|-------------------|
| 0 | Cart position | Low |
| 1 | Cart velocity | Medium |
| 2 | Pole angle | High |
| 3 | Pole angular velocity | **Highest** |

---

## Results

### Dropout Robustness Sweep
The robust agent maintains significantly higher mean reward than the baseline as dropout rate increases. The performance gap widens sharply beyond 30% dropout.

### Sensor Importance
Masking pole angular velocity (dim 3) alone causes the largest performance collapse — more than masking any other single dimension or even cart velocity + position combined.

### Learning Curves
The robust agent converges slightly slower during training (~10–15% more timesteps to plateau) but achieves comparable peak performance on clean observations.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| RL Algorithm | PPO (Stable Baselines3) |
| Environment | CartPole-v1 (Gymnasium) |
| Observation wrapper | Custom `SensorDropoutWrapper` |
| Ablation | Custom `SingleDimDropout` |
| Dashboard | Streamlit |
| Visualisation | Matplotlib |

---

## Why This Matters

Sensor dropout training is a lightweight form of **observation robustness** that maps directly to real-world deployment concerns — sensors fail, observations are noisy, and agents trained only on clean data are fragile. This project quantifies that fragility and demonstrates a simple mitigation.

---

## License

MIT
