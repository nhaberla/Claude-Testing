#!/usr/bin/env python3
"""
Replicates the commercial airliner silhouette from the reference image.
3/4 top-forward perspective: both wings visible, nose upper-right, engines under wings.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# ── helpers ───────────────────────────────────────────────────────────────────
def rot(pts, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    pts = np.asarray(pts, float)
    return np.c_[pts[:,0]*c - pts[:,1]*s, pts[:,0]*s + pts[:,1]*c]

def poly_patch(pts, **kw):
    p = np.asarray(pts, float)
    codes = [Path.MOVETO] + [Path.LINETO]*(len(p)-1) + [Path.CLOSEPOLY]
    verts = np.vstack([p, p[:1]])
    return PathPatch(Path(verts, codes), **kw)

def smooth_patch(pts, **kw):
    """Closed smooth path through pts via per-segment cubic Bezier."""
    from scipy.interpolate import CubicSpline
    pts = np.asarray(pts, float)
    n = len(pts)
    # parameterise by cumulative chord length, ensuring strictly increasing
    d = [0.0]
    for i in range(1, n):
        step = np.linalg.norm(pts[i] - pts[i-1])
        d.append(d[-1] + max(step, 1e-9))
    # close the loop
    close_step = np.linalg.norm(pts[0] - pts[-1])
    d.append(d[-1] + max(close_step, 1e-9))
    d = np.array(d)
    pts_loop = np.vstack([pts, pts[:1]])
    csx = CubicSpline(d, pts_loop[:,0])
    csy = CubicSpline(d, pts_loop[:,1])
    fine = np.linspace(0, d[-2], 500)   # stop before closing duplicate
    xy = np.c_[csx(fine), csy(fine)]
    codes = [Path.MOVETO] + [Path.LINETO]*(len(xy)-1) + [Path.CLOSEPOLY]
    verts = np.vstack([xy, xy[:1]])
    return PathPatch(Path(verts, codes), **kw)

# ── canvas ────────────────────────────────────────────────────────────────────
BG = 'white'
FG = 'black'

fig = plt.figure(figsize=(10, 7), facecolor=BG, dpi=300)
ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-4.0, 4.0)

KW = dict(facecolor=FG, edgecolor='none', zorder=2)

# ── all geometry defined nose→right (+x), upper wing → +y, then rotated ──────
# Final orientation: nose upper-right (~38° above horizontal), matching reference.
TILT = 32   # degrees: nose above horizontal

# ─── FUSELAGE ─────────────────────────────────────────────────────────────────
# Long, narrow, tapered oval — slightly asymmetric (flatter belly).
# Traced clockwise: nose tip → upper edge → tail → lower edge → back.
fus = np.array([
    # Nose
    [ 4.50,  0.00],
    # Upper nose cone
    [ 4.10,  0.25],
    [ 3.20,  0.44],
    # Upper cabin
    [ 1.50,  0.55],
    [ 0.00,  0.58],
    [-1.50,  0.56],
    # Upper tail
    [-3.00,  0.48],
    [-3.80,  0.36],
    # Tail tip
    [-4.40,  0.10],
    [-4.40, -0.08],
    # Lower tail
    [-3.80, -0.30],
    [-3.00, -0.44],
    # Lower belly (slightly flatter)
    [-1.50, -0.52],
    [ 0.00, -0.55],
    [ 1.50, -0.52],
    # Lower nose
    [ 3.20, -0.42],
    [ 4.10, -0.22],
    [ 4.48, -0.02],
])
ax.add_patch(smooth_patch(rot(fus, TILT), **KW))

# ─── UPPER (NEAR) WING ────────────────────────────────────────────────────────
# Large swept-back wing, extends toward upper-left of final image.
# In local frame (+y = up), this is the wing going in +y direction.
wing_up = np.array([
    # Leading edge at fuselage root (forward of wing)
    [ 1.20,  0.56],
    # Leading edge sweeps to tip
    [-0.10,  3.10],
    # Wingtip
    [-0.55,  3.20],
    [-1.10,  3.05],
    # Trailing edge sweeps back
    [-1.60,  2.60],
    [-2.10,  0.56],
    # Trailing edge root at fuselage
])
ax.add_patch(smooth_patch(rot(wing_up, TILT), **KW))

# ─── ENGINE POD — upper wing ──────────────────────────────────────────────────
# Oval nacelle mounted under/forward of the upper wing, ~40 % span.
t = np.linspace(0, 2*np.pi, 200)
# Engine aligned with wing sweep angle (~55° in local frame)
ang_e = np.radians(57)
ea, eb = 0.78, 0.21   # semi-major, semi-minor
ecx, ecy = -0.65, 1.68
eng_up = np.c_[ecx + ea*np.cos(t)*np.cos(ang_e) - eb*np.sin(t)*np.sin(ang_e),
               ecy + ea*np.cos(t)*np.sin(ang_e) + eb*np.sin(t)*np.cos(ang_e)]
ax.add_patch(smooth_patch(rot(eng_up, TILT), **KW))

# ─── LOWER (FAR) WING ─────────────────────────────────────────────────────────
# Foreshortened by perspective: shorter span, same sweep.
wing_dn = np.array([
    [ 1.10, -0.55],
    [-0.05, -2.10],
    [-0.50, -2.18],
    [-0.95, -2.05],
    [-1.50, -1.70],
    [-2.00, -0.55],
])
ax.add_patch(smooth_patch(rot(wing_dn, TILT), **KW))

# ─── ENGINE POD — lower wing ─────────────────────────────────────────────────
ea2, eb2 = 0.62, 0.17
ecx2, ecy2 = -0.52, -1.25
ang_e2 = np.radians(57)
eng_dn = np.c_[ecx2 + ea2*np.cos(t)*np.cos(ang_e2) - eb2*np.sin(t)*np.sin(ang_e2),
               ecy2 + ea2*np.cos(t)*np.sin(ang_e2) + eb2*np.sin(t)*np.cos(ang_e2)]
ax.add_patch(smooth_patch(rot(eng_dn, TILT), **KW))

# ─── VERTICAL TAIL FIN ────────────────────────────────────────────────────────
# Swept triangle, rises from fuselage top at the tail.
vtail = np.array([
    [-2.90,  0.50],   # base leading edge (front)
    [-2.40,  1.70],   # fin tip
    [-3.60,  1.20],   # fin tip trailing
    [-4.10,  0.48],   # base trailing edge (rear)
])
ax.add_patch(smooth_patch(rot(vtail, TILT), **KW))

# ─── HORIZONTAL STABILISERS ───────────────────────────────────────────────────
# Small swept surfaces at the tail — one each side.
hstab_up = np.array([
    [-3.50,  0.50],
    [-3.20,  1.10],
    [-3.60,  1.05],
    [-4.00,  0.50],
])
hstab_dn = np.array([
    [-3.50, -0.44],
    [-3.20, -0.98],
    [-3.60, -0.92],
    [-4.00, -0.44],
])
ax.add_patch(smooth_patch(rot(hstab_up, TILT), **KW))
ax.add_patch(smooth_patch(rot(hstab_dn, TILT), **KW))

# ── save ──────────────────────────────────────────────────────────────────────
out = '/home/user/Claude-Testing/logo.png'
fig.savefig(out, dpi=300, facecolor=BG, bbox_inches='tight', pad_inches=0.15)
plt.close(fig)
print(f'Saved → {out}')
