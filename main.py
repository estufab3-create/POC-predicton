import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
from PIL import Image
# Extra imports for future improvements
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import remove_small_objects, skeletonize, disk, opening
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

# --- Load image ---
img = cv2.imread("sample.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Segmentation step ---
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
gray_eq = clahe.apply(gray)

blur = cv2.GaussianBlur(gray_eq, (7, 7), 0)

block_size = 301  # adjust depending on fiber thickness
adaptive = filters.threshold_local(blur, block_size, offset=3)
binary = blur > adaptive

binary_clean = morphology.remove_small_objects(binary, 15)

labels = measure.label(binary_clean)
regions = measure.regionprops(labels)

# Debug: histogram of widths
widths = [float(r.minor_axis_length) for r in regions]
plt.hist(widths, bins=50, color='steelblue')
plt.xlim(0, 200)
plt.title("Distribution of Fiber Widths (zoomed in)")
plt.xlabel("Width (px)")
plt.ylabel("Count")
plt.show()
print("Largest detected width:", max(widths) if widths else "none")

# --- Classification step with 6 levels ---
thresholds = [80, 100, 200, 250, 500]   # now 5 thresholds → 6 classes
print("Suggested thresholds:", np.percentile)
alpha = 1
weights = {
    'ideal': 1.0,
    'good': 0.9,
    'acceptable': 0.7,
    'poor': 0.4,
    'very poor': 0.2,
    'extremely poor': 0.05
}

colors = {
    'ideal':          (0, 255, 255),   # cyan
    'good':           (0, 255, 0),     # green
    'acceptable':     (255, 255, 0),   # yellow
    'poor':           (255, 165, 0),   # orange
    'very poor':      (255, 0, 0),     # red
    'extremely poor': (128, 0, 128),   # purple
}

# fallback segmentation if needed
if len(regions) == 0:
    print("Debug: fallback segmentation triggered")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    adaptive = filters.threshold_local(blur, 35, offset=10)
    binary = blur > adaptive
    binary_clean = morphology.remove_small_objects(binary, 100)
    selem = morphology.disk(4)  
    binary_sep = morphology.opening(binary_clean, selem)
    labels = measure.label(binary_sep)
    regions = measure.regionprops(labels)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# prepare masks / counters
mask_rgb = np.zeros_like(img_rgb, dtype=np.uint8)
counts = {k: 0 for k in colors.keys()}
pixel_counts = {k: 0 for k in colors.keys()}

# classification loop
for i, r in enumerate(regions):
    try:
        w = float(r.minor_axis_length)
        mask = (labels == r.label)
        if w < thresholds[0]:
            cls = 'ideal'
        elif w < thresholds[1]:
            cls = 'good'
        elif w < thresholds[2]:
            cls = 'acceptable'
        elif w < thresholds[3]:
            cls = 'poor'
        elif w < thresholds[4]:
            cls = 'very poor'
        else:
            cls = 'extremely poor'
        counts[cls] += 1
        pixel_counts[cls] += int(np.count_nonzero(mask))
        mask_rgb[mask] = colors[cls]
    except Exception as e:
        print(f"Warning: region idx={i}, error={e}")

# --- Metrics ---
total_regions = sum(counts.values())
total_pixels = int(np.count_nonzero(labels > 0))

poc_count_ideal = (counts['ideal'] / total_regions * 100) if total_regions > 0 else 0.0
weighted_count_score = (sum(counts[c] * weights[c] for c in counts) / total_regions * 100) if total_regions > 0 else 0.0
weighted_area_score  = (sum(pixel_counts[c] * weights[c] for c in pixel_counts) / total_pixels * 100) if total_pixels > 0 else 0.0
ideal_area_pct = (pixel_counts['ideal'] / total_pixels * 100) if total_pixels > 0 else 0.0

print("=== Classification summary ===")
print(f"Regions detected: {total_regions}")
print("Counts by class:", counts)
print("Pixels by class:", pixel_counts)
print(f"POC (ideal count %) = {poc_count_ideal:.2f}%")
print(f"POC (weighted count) = {weighted_count_score:.2f}%")
print(f"POC (weighted area)  = {weighted_area_score:.2f}%")
print(f"Ideal area % = {ideal_area_pct:.2f}%")

# --- Visualization ---
blended = (img_rgb.astype(np.float32) * (1.0 - alpha) +
           mask_rgb.astype(np.float32) * alpha).astype(np.uint8)

from matplotlib import patches as mpatches
cats = ['ideal','good','acceptable','poor','very poor','extremely poor']
legend_labels = [
    f"ideal (< {thresholds[0]} px)",
    f"good  ({thresholds[0]}–{thresholds[1]} px)",
    f"accpt ({thresholds[1]}–{thresholds[2]} px)",
    f"poor  ({thresholds[2]}–{thresholds[3]} px)",
    f"v.poor ({thresholds[3]}–{thresholds[4]} px)",
    f"ext.poor (> {thresholds[4]} px)"
]
patches = [mpatches.Patch(color=np.array(colors[c])/255.0, label=legend_labels[i]) for i, c in enumerate(cats)]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(blended); axes[1].set_title(f"Overlay — Area-weighted POC {weighted_area_score:.2f}%")
axes[1].axis("off")
axes[1].legend(handles=patches, loc='lower right', framealpha=0.9)

# centroid marks
for r in regions:
    cy, cx = r.centroid
    w = float(r.minor_axis_length)
    if w < thresholds[0]:
        cls = 'ideal'
    elif w < thresholds[1]:
        cls = 'good'
    elif w < thresholds[2]:
        cls = 'acceptable'
    elif w < thresholds[3]:
        cls = 'poor'
    elif w < thresholds[4]:
        cls = 'very poor'
    else:
        cls = 'extremely poor'
    axes[1].scatter(cx, cy, s=10, color=np.array(colors[cls])/255.0, edgecolor='k', linewidth=0.3)

plt.tight_layout()
plt.show()

cv2.imwrite("sample_overlay.png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
print("Saved sample_overlay.png")






