#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

BG  = '#07101E'
FG  = '#FFFFFF'
DPI = 400

fig = plt.figure(figsize=(8, 8), facecolor=BG, dpi=DPI)
ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-1.65, 1.65)
ax.set_ylim(-1.65, 1.65)

R     = 1.0   # globe ring radius
SCALE = 1.08  # plane scale relative to globe
ANGLE = np.radians(37)  # nose points upper-right

def rotate(pts, a):
    pts = np.array(pts, dtype=float)
    c, s = np.cos(a), np.sin(a)
    return np.column_stack([pts[:,0]*c - pts[:,1]*s,
                            pts[:,0]*s + pts[:,1]*c])

# ── Plane silhouette (nose at +x, tail at -x, wings along ±y) ───────────────
# Shape mirrors the reference: broad swept wings, tapered fuselage, twin tail fins.
plane_raw = np.array([
    [ 1.02,  0.00],   # nose tip
    [ 0.36,  0.13],   # upper body at wing-root leading edge
    [ 0.02,  0.83],   # upper wing-tip
    [-0.28,  0.23],   # upper wing trailing edge
    [-0.56,  0.21],   # upper body rear
    [-0.64,  0.42],   # upper tail-fin tip
    [-0.90,  0.00],   # tail tip (center)
    [-0.64, -0.42],   # lower tail-fin tip
    [-0.56, -0.21],   # lower body rear
    [-0.28, -0.23],   # lower wing trailing edge
    [ 0.02, -0.83],   # lower wing-tip
    [ 0.36, -0.13],   # lower body at wing-root leading edge
]) * SCALE

plane = rotate(plane_raw, ANGLE)

# ── Circle ring (globe boundary) ─────────────────────────────────────────────
t = np.linspace(0, 2*np.pi, 3000)
cx, cy = R * np.cos(t), R * np.sin(t)

# ── Orbital swoosh ────────────────────────────────────────────────────────────
# Sweeps CCW from lower-right (~-50°) most of the way around to lower-left (~205°),
# sitting just outside the globe ring.  Rounded caps give it clean tapered ends.
R_sw  = 1.165
t_sw  = np.linspace(np.radians(-52), np.radians(208), 2500)
sx, sy = R_sw * np.cos(t_sw), R_sw * np.sin(t_sw)

# ── Draw: circle ring → plane → swoosh ───────────────────────────────────────
ax.plot(cx, cy, color=FG, linewidth=5.5, antialiased=True, zorder=2)
ax.add_patch(Polygon(plane, closed=True, facecolor=FG, edgecolor='none', zorder=3))
ax.plot(sx, sy, color=FG, linewidth=11.5,
        solid_capstyle='round', antialiased=True, zorder=4)

out = '/home/user/Claude-Testing/logo.png'
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches='tight', pad_inches=0.18)
plt.close(fig)
print(f'Saved → {out}')
