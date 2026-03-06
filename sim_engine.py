import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1600の知性
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

points = project_11d_to_3d() * 120
vel = np.zeros_like(points)

# 背景を少しグレーに寄せるか迷いましたが、やはり漆黒の中で粒子を光らせるのが「情報の創発」に相応しいので、粒子側を強化します。
fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

def update(frame):
    global points, vel
    ax.clear()
    ax.set_axis_off()
    
    # 物理法則
    dist = np.linalg.norm(points, axis=1)[:, None] + 0.5
    force = -3.0 * points / (dist**2.5) # 引き込みを少し強めに
    vel += force
    vel *= 0.94
    points += vel
    
    # ビジュアル強化：速度に応じてサイズと色を変える
    speed = np.linalg.norm(vel, axis=1)
    
    # 彩度を高めるために 'hsv' や 'cyan' 系のグラデーションを使用
    # s（サイズ）を大きくして、alpha（透明度）を調整
    ax.scatter(points[:,0], points[:,1], points[:,2], 
               c=speed, 
               cmap='cyan_magenta', # カスタム的に明るい色を選択（あるいは 'winter' や 'spring'）
               s=speed * 5 + 2,    # 動いている粒子ほど大きく輝く
               alpha=0.8, 
               edgecolors='none')
    
    # 視点の回転
    ax.view_init(elev=25, azim=frame * 4)
    return ax,

print("Generating Brighter Universe...")
# cmapを 'winter' (青〜緑) にすると、よりサイバーで明るい印象になります
ani = FuncAnimation(fig, update, frames=TIME_STEPS, interval=50)
# dpiを上げると鮮明になります
ani.save('suzuki_universe_evolution.mp4', writer='ffmpeg', fps=30, dpi=120)
print("Bright Evolution Complete.")
