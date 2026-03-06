import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# 1600の記事＝知性の粒子
N = 1600
TIME_STEPS = 120
PHI = (1 + np.sqrt(5)) / 2

# 11次元から3次元への射影
def project_11d_to_3d():
    p11 = np.random.normal(0, 1, (N, 11))
    proj_mat = np.array([
        [1, PHI, 1/PHI, PHI**2, 1, PHI, 1/PHI, PHI**2, 1, PHI, 1/PHI],
        [PHI, 1/PHI, PHI**2, 1, PHI, 1/PHI, PHI**2, 1, PHI, 1/PHI, PHI**2],
        [1/PHI, PHI**2, 1, PHI, 1/PHI, PHI**2, 1, PHI, 1/PHI, PHI**2, 1]
    ])
    p3 = p11 @ proj_mat.T
    return p3 / (np.linalg.norm(p3, axis=1)[:, None] + 1e-9)

points = project_11d_to_3d() * 100
vel = np.zeros_like(points)

fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

def update(frame):
    global points, vel
    ax.clear()
    ax.set_axis_off()
    
    # 中心への引力（情報の集束）
    dist = np.linalg.norm(points, axis=1)[:, None] + 0.5
    force = -2.0 * points / (dist**3)
    vel += force
    vel *= 0.95 # 摩擦
    points += vel
    
    # 描画
    ax.scatter(points[:,0], points[:,1], points[:,2], 
               c=np.linalg.norm(vel, axis=1), cmap='cool', s=2, alpha=0.6)
    ax.view_init(elev=20, azim=frame * 2)
    return ax,

ani = FuncAnimation(fig, update, frames=TIME_STEPS, interval=50)
# GitHub Actions上で保存
ani.save('suzuki_universe_evolution.mp4', writer='ffmpeg', fps=30, dpi=100)
