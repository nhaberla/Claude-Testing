#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ── Canvas ────────────────────────────────────────────────────────────────────
BG   = '#07080F'
INK  = '#FFFFFF'
DPI  = 400
W, H = 8, 8          # inches  →  3200 × 3200 px

fig = plt.figure(figsize=(W, H), facecolor=BG, dpi=DPI)
ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
ax.set_aspect('equal')
ax.axis('off')

# Shift composition slightly right so the tail fin and globe are visually centred
OFFSET_X = 0.28

PAD_X = 2.0
PAD_Y = 1.6
ax.set_xlim(-PAD_X + OFFSET_X, PAD_X + OFFSET_X)
ax.set_ylim(-PAD_Y, PAD_Y)

# ── Geometry ──────────────────────────────────────────────────────────────────
R  = 1.0
LW = 5.2       # refined stroke weight

# SEGMENT 1 – orbital arc
# 300° CCW from 60° → arrives at 60°+300° = 360°=0° = (1,0).
# Path: upper-right → top → left → bottom → right.
# Tangent at arrival (1,0) is straight up (+y) → clean 90° pivot into fuselage.
theta = np.linspace(np.radians(60), np.radians(60 + 300), 3000)
arc_x = R * np.cos(theta) + OFFSET_X
arc_y = R * np.sin(theta)

# SEGMENT 2 – fuselage / diameter
# From (1,0)+offset going LEFT, crossing the full diameter, and extending
# 0.62 beyond the circle on the left to give the tail fin a clear base.
TAIL_ROOT_X = -(R + 0.62) + OFFSET_X
diam_x = np.linspace(R + OFFSET_X, TAIL_ROOT_X, 500)
diam_y = np.zeros(500)

# SEGMENT 3 – tail fin (cubic Bézier)
# Base at TAIL_ROOT_X, 0. Sweeps rearward (further left) as it rises, then the
# leading edge rakes forward at the crown — a modern swept stabiliser profile.
def cubic_bezier(P0, P1, P2, P3, t):
    t = t[:, None]
    return (1-t)**3*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3

t_fin = np.linspace(0, 1, 1000)
# Tighter, more elegant proportions on the fin
P0 = np.array([TAIL_ROOT_X,       0.00])
P1 = np.array([TAIL_ROOT_X-0.18,  0.22])  # rearward lean on entry
P2 = np.array([TAIL_ROOT_X-0.20,  0.58])  # continued climb
P3 = np.array([TAIL_ROOT_X+0.22,  0.90])  # tip rakes firmly forward

fin = cubic_bezier(P0, P1, P2, P3, t_fin)

# ── Assemble single continuous path ───────────────────────────────────────────
px = np.concatenate([arc_x, diam_x[1:], fin[1:, 0]])
py = np.concatenate([arc_y, diam_y[1:], fin[1:, 1]])

# ── Render ────────────────────────────────────────────────────────────────────
# Three-layer rendering: outer glow → soft halo → crisp ink line
# The glow gives the line presence without compromising the minimalist aesthetic.
ax.plot(px, py,
        color='#D0D8FF', linewidth=LW * 5.5, alpha=0.030,
        solid_capstyle='round', solid_joinstyle='round',
        antialiased=True, zorder=1)

ax.plot(px, py,
        color='#E8EEFF', linewidth=LW * 2.2, alpha=0.10,
        solid_capstyle='round', solid_joinstyle='round',
        antialiased=True, zorder=2)

ax.plot(px, py,
        color=INK, linewidth=LW,
        solid_capstyle='round', solid_joinstyle='round',
        antialiased=True, zorder=3)

# ── Export ────────────────────────────────────────────────────────────────────
out = '/home/user/Claude-Testing/logo.png'
fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches='tight', pad_inches=0.18)
plt.close(fig)
print(f'Saved → {out}')
