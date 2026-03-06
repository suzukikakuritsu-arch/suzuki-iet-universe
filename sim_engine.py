import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1600の粒子
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
    
    # 物理法則（中心への引き込み）
    dist = np.linalg.norm(points, axis=1)[:, None] + 0.5
    force = -3.5 * points / (dist**2.8) 
    vel += force
    vel *= 0.93
    points += vel
    
    speed = np.linalg.norm(vel, axis=1)
    
    # --- 輝きの演出 ---
    # 1. 背後のぼんやりした光（大きなサイズ、高透明度）
    ax.scatter(points[:,0], points[:,1], points[:,2], 
               c=speed, cmap='Wistia', s=speed*15, alpha=0.1, edgecolors='none')
    
    # 2. 中心の実体（小さなサイズ、低透明度）
    ax.scatter(points[:,0], points[:,1], points[:,2], 
               c=speed, cmap='Wistia', s=speed*3 + 1, alpha=0.8, edgecolors='none')
    
    ax.view_init(elev=20, azim=frame * 3)
    return ax,

print("Generating Brighter Universe...")
ani = FuncAnimation(fig, update, frames=TIME_STEPS, interval=50)
ani.save('suzuki_universe_evolution.mp4', writer='ffmpeg', fps=30, dpi=100)
print("Complete.")
