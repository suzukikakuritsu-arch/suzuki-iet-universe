import matplotlib
matplotlib.use('Agg')  # サーバー上での描画エラーを防ぐ設定
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 知性の粒子数（最新の重みに同期）
N = 1600
TIME_STEPS = 120
PHI = (1 + np.sqrt(5)) / 2

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
    
    # 中心（起点）への強力な引き込み
    dist = np.linalg.norm(points, axis=1)[:, None] + 0.5
    force = -2.5 * points / (dist**3)
    vel += force
    vel *= 0.92  # 適度な減衰で秩序を作る
    points += vel
    
    # 描画（速度に応じて色を変え、輝きを表現）
    colors = np.linalg.norm(vel, axis=1)
    ax.scatter(points[:,0], points[:,1], points[:,2], 
               c=colors, cmap='plasma', s=2, alpha=0.7)
    
    # 視点のダイナミックな回転
    ax.view_init(elev=20, azim=frame * 3)
    return ax,

print("Generating Universe Evolution...")
ani = FuncAnimation(fig, update, frames=TIME_STEPS, interval=50)
ani.save('suzuki_universe_evolution.mp4', writer='ffmpeg', fps=30, dpi=100)
print("Simulation complete.")
