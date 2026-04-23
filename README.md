# Deep Convolutional Q-Learning for Ms. Pac-Man

A Double DQN agent trained to play Ms. Pac-Man using deep convolutional networks, built with PyTorch and Gymnasium.

The agent solved the environment (**average score ≥ 1500 over 100 consecutive episodes**) in **961 episodes**.

## Training curve

| Episode | Avg Score (100 ep) |
|---------|-------------------|
| 100     | 296.10            |
| 200     | 495.20            |
| 300     | 626.20            |
| 400     | 726.60            |
| 500     | 876.50            |
| 600     | 999.10            |
| 700     | 1199.80           |
| 800     | 1171.40           |
| 900     | 1273.40           |
| 1000    | 1289.20           |
| **1061**| **1503.80 ✓**     |

## Architecture

**Input:** single grayscale frame resized to 84×84 (standard DQN preprocessing)

**Network:**
```
Conv2d(1, 32, kernel=8, stride=4)  → BatchNorm → ReLU   # 20×20
Conv2d(32, 64, kernel=4, stride=2) → BatchNorm → ReLU   # 9×9
Conv2d(64, 64, kernel=3, stride=1) → BatchNorm → ReLU   # 7×7
Flatten → Linear(3136, 512) → ReLU
Linear(512, 9)  # 9 discrete actions
```

## Key design choices

| Component | Choice | Reason |
|-----------|--------|--------|
| Preprocessing | Grayscale 84×84 | ~7× less data per frame vs RGB 128×128; color carries no useful signal in Pac-Man |
| Algorithm | Double DQN | local network selects action, target network evaluates it — reduces Q-value overestimation |
| Target network | Hard update every 1000 steps | stable learning target; updating too frequently causes oscillations |
| Learning | Every 4 environment steps | reduces correlation between consecutive updates |
| Replay buffer | 100 000 transitions | more diverse sampling than the original 10 000 |

## Requirements

```
gymnasium[atari]
ale-py
torch
torchvision
Pillow
imageio
numpy
```

Install with:
```bash
pip install gymnasium "gymnasium[atari,accept-rom-license]" torch torchvision pillow imageio numpy ale-py
```

## Usage

Open `Deep_Convolutional_Q_Learning_for_Pac_Man_v2.ipynb` and run all cells in order.  
A pre-trained checkpoint is available at `checkpoint.pth`.  
A sample gameplay video is saved as `video.mp4` after running Part 3.

## Environment

- **Library:** Gymnasium + ALE  
- **Env ID:** `ALE/MsPacman-v5`  
- **Actions:** 9 (reduced action space)  
- **Observation:** 210×160×3 RGB (preprocessed to 84×84 grayscale)
