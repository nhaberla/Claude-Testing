#!/usr/bin/env python3
"""
Compare two logo images at pixel level.
Resizes both to the same dimensions before comparison.
Usage: python3 compare_logos.py <generated> <target>
"""
import sys
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_GEN    = '/home/user/Claude-Testing/logo.png'
DEFAULT_TARGET = '/home/user/Claude-Testing/target.png'

gen_path    = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GEN
target_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TARGET

# ── load & normalise ──────────────────────────────────────────────────────────
def load_gray(path):
    img = Image.open(path).convert('L')   # greyscale
    return img

gen_img    = load_gray(gen_path)
target_img = load_gray(target_path)

print(f"Generated size : {gen_img.size}")
print(f"Target size    : {target_img.size}")

# Resize generated to match target (or pick a common size)
W, H = target_img.size
gen_resized = gen_img.resize((W, H), Image.LANCZOS)

gen_arr    = np.array(gen_resized,    dtype=np.float32)
target_arr = np.array(target_img, dtype=np.float32)

# Build a mask to exclude border lines and the Shutterstock watermark rows
# so they don't pollute the metrics (these pixels are not part of the artwork).
row_mask = np.ones(H, dtype=bool)
for r in range(H):
    row_ink = np.where(target_arr[r] < 128)[0]
    if len(row_ink) == 0:
        continue
    span = int(row_ink[-1]) - int(row_ink[0])
    if span >= W - 4:          # full-width border line
        row_mask[r] = False
    if r > int(0.65 * H):      # below plane (watermark region for 260×280)
        row_mask[r] = False
PIXEL_MASK = row_mask[:, None] * np.ones(W, dtype=bool)   # broadcast to 2D

# Flat views used for scalar metrics only
gen_flat    = gen_arr[PIXEL_MASK]
target_flat = target_arr[PIXEL_MASK]

# ── metrics (computed on masked region only, watermark/border excluded) ───────
diff     = gen_flat - target_flat
abs_diff = np.abs(diff)

mse  = float(np.mean(diff ** 2))
rmse = float(np.sqrt(mse))
mae  = float(np.mean(abs_diff))

nrmse = rmse / 255.0
nmae  = mae  / 255.0

tolerance  = 10
within_tol = float(np.mean(abs_diff <= tolerance)) * 100

try:
    from skimage.metrics import structural_similarity as ssim
    ssim_score = ssim(np.array(gen_resized, dtype=np.float32),
                      np.array(target_img,  dtype=np.float32), data_range=255)
except ImportError:
    ssim_score = None

gen_ink    = gen_flat    < 128
target_ink = target_flat < 128
intersection = np.logical_and(gen_ink, target_ink).sum()
union        = np.logical_or( gen_ink, target_ink).sum()
iou          = intersection / union if union > 0 else 0.0

gen_only    = np.logical_and(gen_ink,    ~target_ink).sum()
target_only = np.logical_and(target_ink, ~gen_ink   ).sum()
both        = intersection

print()
print("═══════════════════════════════════════")
print("  PIXEL-LEVEL COMPARISON REPORT")
print("═══════════════════════════════════════")
print(f"  Comparison size  : {W} × {H} px")
print()
print(f"  MSE              : {mse:.1f}   (0 = identical; max ~65025)")
print(f"  RMSE             : {rmse:.2f}  (pixel grey-level units, 0-255)")
print(f"  Normalised RMSE  : {nrmse*100:.2f}%")
print(f"  MAE              : {mae:.2f}  (mean absolute pixel error)")
print(f"  Pixels within ±10: {within_tol:.1f}%")
if ssim_score is not None:
    print(f"  SSIM             : {ssim_score:.4f}  (1 = perfect match)")
print()
print(f"  Ink (dark) pixel overlap:")
print(f"    IoU              : {iou*100:.1f}%  (Intersection over Union)")
print(f"    In both          : {both:,} px")
print(f"    Generated only   : {gen_only:,} px")
print(f"    Target only      : {target_only:,} px")
print("═══════════════════════════════════════")

# ── visualisation ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Logo Comparison", fontsize=14, fontweight='bold')

axes[0].imshow(gen_resized,    cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Generated (resized)')
axes[0].axis('off')

axes[1].imshow(target_img, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Target')
axes[1].axis('off')

abs_diff_2d = np.abs(np.array(gen_resized, dtype=np.float32) -
                     np.array(target_img,  dtype=np.float32))
im = axes[2].imshow(abs_diff_2d, cmap='hot', vmin=0, vmax=255)
axes[2].set_title(f'Absolute diff  (MAE={mae:.1f})')
axes[2].axis('off')
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

gen_ink_2d    = np.array(gen_resized, dtype=np.float32) < 128
target_ink_2d = np.array(target_img,  dtype=np.float32) < 128
overlap_rgb = np.ones((*gen_ink_2d.shape, 3), dtype=np.float32)
overlap_rgb[gen_ink_2d & ~target_ink_2d] = [1, 0.2, 0.2]
overlap_rgb[target_ink_2d & ~gen_ink_2d] = [0.2, 0.2, 1.0]
overlap_rgb[gen_ink_2d & target_ink_2d]  = [0.1, 0.1, 0.1]
axes[3].imshow(overlap_rgb)
axes[3].set_title(f'Ink overlap  IoU={iou*100:.1f}%\n'
                  f'■ both  ■ gen-only  ■ target-only')
axes[3].axis('off')

out_path = '/home/user/Claude-Testing/comparison.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nDiff image saved → {out_path}")
