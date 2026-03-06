import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- 定数設定 ---
PHI = (1 + np.sqrt(5)) / 2
N_PARTICLES = 1600 # 1600の記事に対応
TIME_STEPS = 100   # 動画の長さ（フレーム数）
G = 1.0
DT = 0.015

# --- 11次元からの黄金比射影 ---
def get_projected_3d():
    p11 = np.random.normal(0, 1, (N_PARTICLES, 11))
    # 黄金比を用いた射影行列
    P = np.array([
        [1, PHI, 1/PHI, PHI**2, 1, PHI, 1/PHI, PHI**2, 1, PHI, 1/PHI],
        [PHI, 1/PHI, PHI**2, 1, PHI, 1/PHI, PHI**2, 1, PHI, 1/PHI, PHI**2],
        [1/PHI, PHI**2, 1, PHI, 1/PHI, PHI**2, 1, PHI, 1/PHI, PHI**2, 1]
    ])
    projected = p11 @ P.T
    return projected / (np.linalg.norm(projected, axis=1)[:, None] + 1e-9)

points = get_projected_3d() * 100
velocity = np.zeros_like(points)

# --- アニメーション設定 ---
fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')
ax.grid(False)
ax.set_axis_off()

scatter = ax.scatter([], [], [], c='cyan', s=2, alpha=0.6)

def update(frame):
    global points, velocity
    # 簡易重力計算（中心への集束）
    dist = np.linalg.norm(points, axis=1)[:, None] + 0.1
    force = -G * points / (dist**3)
    velocity += force * DT
    points += velocity * DT
    
    # プロット更新
    ax.clear()
    ax.set_axis_off()
    ax.set_facecolor('black')
    
    # 粒子の描画
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=np.linalg.norm(velocity, axis=1), cmap='winter', s=3, alpha=0.8)
    
    # 視点をゆっくり回転させる
    ax.view_init(elev=20, azim=frame * 0.5)
    
    print(f"Frame {frame}/{TIME_STEPS} generating...")
    return scatter,

# 動画保存の設定
# ※FFMpegがインストールされている必要があります
ani = FuncAnimation(fig, update, frames=TIME_STEPS, interval=50)
ani.save('suzuki_universe_evolution.mp4', writer='ffmpeg', fps=30, dpi=150)

print("Movie saved as suzuki_universe_evolution.mp4")
