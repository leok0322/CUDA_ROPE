"""
Visualize complex vector rotation z -> z * e^(iφ) in Cartesian and Polar coordinates.
This is the core operation in Rotary Position Embedding (RoPE).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

# ── Output directory ─────────────────────────────────────────────────────────
# 路径层级：src/script/ -> src/ -> CUDA_ROPE/（项目根目录）
_script_dir   = os.path.dirname(os.path.abspath(__file__))   # 当前脚本所在目录: src/script/
_project_root = os.path.dirname(os.path.dirname(_script_dir))  # 上两级到项目根目录: CUDA_ROPE/
OUTPUT_DIR    = os.path.join(_project_root, 'plot_output')   # 图片输出目录: CUDA_ROPE/plot_output/
os.makedirs(OUTPUT_DIR, exist_ok=True)                        # 目录不存在时自动创建

# ── Parameters ──────────────────────────────────────────────────────────────
z_real, z_imag = 2.0, 1.5          # original complex vector z = a + bi
phi = np.pi / 4                    # rotation angle φ (45°)

z     = complex(z_real, z_imag)
rot   = np.exp(1j * phi)           # e^(iφ)
z_rot = z * rot                    # z · e^(iφ)

# Derived quantities
r      = abs(z)
theta  = np.angle(z)
r_rot  = abs(z_rot)                # same magnitude
theta_rot = np.angle(z_rot)        # theta + phi

print(f"z         = {z.real:.4f} + {z.imag:.4f}i   |z| = {r:.4f}   arg(z) = {np.degrees(theta):.2f}°")
print(f"e^(iφ)    = {rot.real:.4f} + {rot.imag:.4f}i   φ  = {np.degrees(phi):.2f}°")
print(f"z·e^(iφ)  = {z_rot.real:.4f} + {z_rot.imag:.4f}i   |z·e^(iφ)| = {r_rot:.4f}   arg = {np.degrees(theta_rot):.2f}°")

# ── Figure layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6.5))
fig.suptitle(r"Complex Rotation: $z \cdot e^{i\varphi}$  —  RoPE Core Operation",
             fontsize=15, fontweight='bold', y=1.01)

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
ax_cart = fig.add_subplot(gs[0])
ax_pol  = fig.add_subplot(gs[1], projection='polar')

COLORS = {'z': '#2196F3', 'zrot': '#F44336', 'angle': '#4CAF50', 'arc': '#9C27B0'}


# ════════════════════════════════════════════════════════════════════════════
# Left: Cartesian coordinate system
# ════════════════════════════════════════════════════════════════════════════
ax = ax_cart
lim = r * 1.6

ax.set_xlim(-lim * 0.3, lim)
ax.set_ylim(-lim * 0.3, lim)
ax.set_aspect('equal')
ax.axhline(0, color='k', lw=0.8)
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('Real axis', fontsize=11)
ax.set_ylabel('Imaginary axis', fontsize=11)
ax.set_title('Cartesian Coordinate System', fontsize=12, pad=10)
ax.grid(True, linestyle='--', alpha=0.4)

def draw_arrow(ax, x, y, color, lw=2.2, label=''):
    ax.annotate('', xy=(x, y), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=18))
    if label:
        offset = np.array([x, y]) * 0.08 + np.array([0.06, 0.06])
        ax.text(x + offset[0], y + offset[1], label, color=color,
                fontsize=12, fontweight='bold')

# Unit circle (dashed)
theta_circ = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta_circ), np.sin(theta_circ),
        color='gray', lw=0.7, linestyle=':', alpha=0.6)
ax.text(1.05, 0.05, '1', color='gray', fontsize=9)

# z vector
draw_arrow(ax, z.real, z.imag, COLORS['z'], label=r'$z$')
ax.plot([z.real, z.real], [0, z.imag], color=COLORS['z'], lw=0.8, linestyle='--', alpha=0.5)
ax.plot([0, z.real],      [z.imag, z.imag], color=COLORS['z'], lw=0.8, linestyle='--', alpha=0.5)
ax.text(z.real / 2 - 0.15, -0.18, f'a={z.real:.2f}', color=COLORS['z'], fontsize=9)
ax.text(z.real + 0.08, z.imag / 2, f'b={z.imag:.2f}', color=COLORS['z'], fontsize=9)

# z·e^(iφ) vector
draw_arrow(ax, z_rot.real, z_rot.imag, COLORS['zrot'], label=r"$z \cdot e^{i\varphi}$")

# Arc showing φ angle from z to z_rot
arc_r = r * 0.45
arc_theta = np.linspace(theta, theta_rot, 60)
ax.plot(arc_r * np.cos(arc_theta), arc_r * np.sin(arc_theta),
        color=COLORS['angle'], lw=1.8)
mid_arc = (theta + theta_rot) / 2
ax.text(arc_r * 1.15 * np.cos(mid_arc), arc_r * 1.15 * np.sin(mid_arc),
        r'$\varphi$', color=COLORS['angle'], fontsize=13)

# Arc showing θ angle from real axis to z
arc_r2 = r * 0.28
arc_theta2 = np.linspace(0, theta, 60)
ax.plot(arc_r2 * np.cos(arc_theta2), arc_r2 * np.sin(arc_theta2),
        color='gray', lw=1.2, linestyle='--')
ax.text(arc_r2 * 1.3, arc_r2 * 0.4, r'$\theta$', color='gray', fontsize=11)

# |z| = r annotation
mid = np.array([z.real, z.imag]) / 2
perp = np.array([-z.imag, z.real]) / r * 0.18
ax.text(mid[0] + perp[0], mid[1] + perp[1], f'r={r:.2f}',
        color=COLORS['z'], fontsize=9, ha='center',
        rotation=np.degrees(theta))

# Legend
legend_handles = [
    mpatches.Patch(color=COLORS['z'],     label=r'$z = a + bi = r e^{i\theta}$'),
    mpatches.Patch(color=COLORS['zrot'],  label=r'$z \cdot e^{i\varphi} = r e^{i(\theta+\varphi)}$'),
    mpatches.Patch(color=COLORS['angle'], label=rf'rotation $\varphi={np.degrees(phi):.0f}°$'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9)


# ════════════════════════════════════════════════════════════════════════════
# Right: Polar coordinate system
# ════════════════════════════════════════════════════════════════════════════
ax2 = ax_pol
ax2.set_title('Polar Coordinate System', fontsize=12, pad=18)
ax2.set_ylim(0, r * 1.45)
ax2.set_rlabel_position(22.5)
ax2.tick_params(labelsize=8)
ax2.grid(True, alpha=0.35)

# z in polar
ax2.annotate('', xy=(theta, r), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color=COLORS['z'], lw=2.2,
                             mutation_scale=18))
ax2.text(theta + 0.08, r * 1.08, r'$z$', color=COLORS['z'], fontsize=13, fontweight='bold')

# z·e^(iφ) in polar
ax2.annotate('', xy=(theta_rot, r_rot), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color=COLORS['zrot'], lw=2.2,
                             mutation_scale=18))
ax2.text(theta_rot + 0.06, r_rot * 1.08, r"$z \cdot e^{i\varphi}$",
         color=COLORS['zrot'], fontsize=13, fontweight='bold')

# Arc between the two vectors
arc_r_p = r * 0.55
arc_t_p = np.linspace(theta, theta_rot, 60)
ax2.plot(arc_t_p, [arc_r_p] * 60, color=COLORS['angle'], lw=2.0)
ax2.text((theta + theta_rot) / 2 + 0.02, arc_r_p * 1.18,
         r'$\varphi$', color=COLORS['angle'], fontsize=13)

# r annotation
ax2.text(theta - 0.25, r * 0.52, f'r={r:.2f}', color=COLORS['z'],
         fontsize=9, ha='center')

# Key formula as text box
formula = (
    f"$z = {z.real:.2f} + {z.imag:.2f}i$\n"
    f"$|z| = r = {r:.3f}$\n"
    f"$\\arg(z) = \\theta = {np.degrees(theta):.1f}°$\n\n"
    f"$\\varphi = {np.degrees(phi):.1f}°$\n\n"
    f"$z \\cdot e^{{i\\varphi}} = {z_rot.real:.3f} + {z_rot.imag:.3f}i$\n"
    f"$|z \\cdot e^{{i\\varphi}}| = {r_rot:.3f}$ (unchanged)\n"
    f"$\\arg = \\theta + \\varphi = {np.degrees(theta_rot):.1f}°$"
)
fig.text(0.512, 0.02, formula, fontsize=9.5,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#f5f5f5', edgecolor='#bdbdbd', alpha=0.95),
         va='bottom', ha='center', family='monospace')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'rope_rotation.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")
plt.show()
