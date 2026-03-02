import os
import numpy as np
import pandas as pd
import tifffile as tiff

from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_fill_holes
from skimage import filters
from skimage.morphology import (remove_small_objects, remove_small_holes,
                                 disk, binary_opening, binary_dilation, h_maxima)
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

# ============================================================
# WHAT CHANGED AND WHY
# ============================================================
# The main improvement is in nuclei seed detection.
# Original approach: find ALL local maxima of the distance transform
#   → fails for touching nuclei because the distance map has a broad
#     plateau between two adjacent cells, producing either one merged
#     region or spurious seeds at the saddle point.
#
# New approach: h-maxima transform before peak detection
#   → only keeps maxima that are at least H pixels "taller" than their
#     surroundings. This suppresses the shallow saddle points between
#     touching nuclei while preserving the deep peaks at true cell centers.
#
# Additionally:
#   - Stronger gaussian smoothing on the distance map (sigma 2.0 vs 1.0)
#     to merge spurious peaks within the same nucleus
#   - Marker-controlled watershed with compactness to produce rounder,
#     more biologically realistic nuclear boundaries
#   - Optional: shape filter to discard merged objects that are too large
#     or have unusual elongation (eccentricity)
# ============================================================

# ---------------- Global parameters ----------------
PIXEL_SIZE_UM    = 0.0823496   # µm/pixel
MIN_AREA_UM2     = 100.0       # min object area in µm²
MAX_AREA_UM2     = 5000.0  # was 2000.0 — fibroblasts can be large

# Channel indices (0-based)
ACTIN_CHANNEL    = 0
MYOSIN_CHANNEL   = 1
PMYOSIN_CHANNEL  = 2
NUCLEI_CHANNEL   = 3

# Thresholding
OTSU_SCALE_NUC   = 0.25    # was 0.40 — more permissive to catch dim nuclei
OTSU_SCALE_ACT   = 0.20

# Distance-watershed parameters
GAUSS_SIGMA_PX   = 2.0         # INCREASED from 1.0: smoother distance map → cleaner peaks
PEAK_MIN_DIST_UM = 5.0         # minimum spacing between nuclei seeds (µm)
EXCLUDE_BORDER   = False

# NEW: h-maxima parameter
# H controls how "prominent" a peak must be to count as a nucleus center.
# Too small → spurious seeds between touching cells
# Too large → under-segmentation (misses real cells)
# Start with H = 3.0 and tune based on your typical nuclear size
H_MAXIMA_H       = 2.0     # was 3.0 — more sensitive seed detection

# NEW: shape filters
MAX_ECCENTRICITY = 0.97    # was 0.90 — fibroblasts are naturally elongated

# Output / debug
PLOT_DEBUG       = True
SAVE_LABEL_TIFS  = False
OUTPUT_DIR       = None


# ---------------- Helpers ----------------
def _extract_channel_plane(img, channel_index):
    """
    Return a 2D plane for the requested channel.
    Handles (Y,X), (C,Y,X), (Z,Y,X), (Z,C,Y,X) and (C,Z,Y,X).
    Max-projects over Z if present.
    """
    if img.ndim == 2:
        return img.astype(np.float32)

    if img.ndim == 3:
        a, b, c = img.shape
        if a <= 8:          # (C, Y, X)
            return img[channel_index].astype(np.float32)
        else:               # (Z, Y, X)
            return np.max(img, axis=0).astype(np.float32)

    if img.ndim == 4:
        dims = sorted(list(enumerate(img.shape)), key=lambda t: t[1], reverse=True)
        (iy, _), (ix, _) = dims[:2]
        rem = [ax for ax in range(4) if ax not in (iy, ix)]
        a0, a1 = rem
        if img.shape[a0] <= 8 and img.shape[a1] > 8:
            c_ax, z_ax = a0, a1
        elif img.shape[a1] <= 8 and img.shape[a0] > 8:
            c_ax, z_ax = a1, a0
        else:
            z_ax, c_ax = 0, 1
        arr = np.moveaxis(img, (c_ax, z_ax, iy, ix), (0, 1, 2, 3))
        cimg = arr[channel_index]
        return np.max(cimg, axis=0).astype(np.float32)

    raise AssertionError("Unsupported image dimensionality.")


def _min_area_to_px(min_area_um2, pixel_size_um):
    px_area = pixel_size_um ** 2
    return int(max(1, np.round(min_area_um2 / px_area)))


def _binary_clean(mask, min_area_px):
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=min_area_px)
    return mask


def _detect_seeds_h_maxima(dist_map_s, mask, pixel_size_um,
                             peak_min_dist_um, h, exclude_border):
    """
    NEW seed detection strategy using h-maxima transform.

    Mathematical explanation:
    -------------------------
    The h-maxima transform suppresses all maxima in the distance map
    whose "height" above their surroundings is less than h pixels.

    Formally, for a grayscale image f, the h-maxima transform is:
        HMAX_h(f) = f - R_f^(f-h)
    where R_f^(f-h) is the morphological reconstruction of (f-h) under f.

    In practice: a local maximum at position p is kept only if
        dist_map(p) - dist_map(saddle_point) >= h
    where saddle_point is the lowest point on the highest path to any
    other local maximum.

    This means:
    - Two touching nuclei → the saddle point between them is shallow
      → h-maxima suppresses it → one seed per nucleus ✅
    - Isolated nucleus → single deep peak → always kept ✅
    - Noise bumps → shallow → suppressed ✅
    """
    # Step 1: h-maxima transform — suppresses shallow peaks
    hmax = h_maxima(dist_map_s, h=h)

    # Step 2: mask to valid foreground region only
    hmax = hmax * mask

    # Step 3: label connected regions in hmax as individual seed candidates
    hmax_labeled = label(hmax)

    # Step 4: from each labeled region take the single highest point
    # This prevents one broad h-maxima region from generating multiple seeds
    min_dist_px = max(1, int(np.round(peak_min_dist_um / pixel_size_um)))
    coords = peak_local_max(
        dist_map_s,
        min_distance=min_dist_px,
        labels=hmax_labeled,        # restrict peaks to h-maxima regions
        exclude_border=exclude_border
    )
    return coords


def _centroids_from_labels(lbl_img):
    seeds = []
    rps = regionprops(lbl_img)
    for rp in rps:
        y, x = rp.centroid
        seeds.append((rp.label, int(round(y)), int(round(x))))
    seeds.sort(key=lambda t: t[0])
    return seeds


# ---------------- Nuclei (DAPI) segmentation ----------------
def segment_nuclei(plane, pixel_size_um, min_area_um2, max_area_um2,
                   otsu_scale, gauss_sigma_px, exclude_border,
                   peak_min_dist_um, h_maxima_h,
                   max_eccentricity, plot=False, fname=""):
    """
    Improved nuclei segmentation with h-maxima seed detection.
    Key changes vs original:
      1. h-maxima transform for robust seed placement on touching nuclei
      2. Stronger gaussian smoothing on distance map
      3. Post-watershed shape filtering (area + eccentricity)
    """
    # --- Threshold ---
    thr = filters.threshold_otsu(plane) * otsu_scale
    mask = plane > thr

    min_area_px = _min_area_to_px(min_area_um2, pixel_size_um)
    max_area_px = _min_area_to_px(max_area_um2, pixel_size_um)
    mask = _binary_clean(mask, min_area_px)

    # --- Distance transform + smoothing ---
    # The distance transform assigns to each foreground pixel its distance
    # to the nearest background pixel. The local maxima of this map are
    # the most "interior" points of each nucleus — ideal seed locations.
    dist = distance_transform_edt(mask)
    dist_s = gaussian_filter(dist, sigma=gauss_sigma_px)

    # --- NEW: h-maxima seed detection ---
    coords = _detect_seeds_h_maxima(
        dist_s, mask, pixel_size_um,
        peak_min_dist_um, h_maxima_h, exclude_border
    )

    if len(coords) == 0:
        print(f"  WARNING: No seeds found in {fname}, skipping.")
        return np.zeros_like(plane, dtype=np.int32), []

    # --- Build markers and run watershed ---
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    markers = label(markers)

    # Watershed on negative distance map (we want to fill from peaks downward)
    # compactness > 0 penalizes irregular boundaries → rounder nuclei
    nuc_labels = watershed(-dist_s, markers=markers, mask=mask, compactness=0.1)

    # --- NEW: Post-watershed shape filtering ---
    # Remove objects that are too large (likely merged nuclei) or too elongated
    props = regionprops(nuc_labels)
    filtered_labels = np.zeros_like(nuc_labels)
    kept = 0
    for rp in props:
        area_px = rp.area
        ecc = rp.eccentricity
        if area_px <= max_area_px and ecc <= max_eccentricity:
            filtered_labels[nuc_labels == rp.label] = rp.label
            kept += 1

    # Re-label consecutively after filtering
    nuc_labels = label(filtered_labels > 0)
    print(f"  {fname}: {len(coords)} seeds → {kept} nuclei after shape filter")

    # --- Measurements ---
    px_area = pixel_size_um ** 2
    props = regionprops(nuc_labels, intensity_image=plane)
    rows = []
    for rp in props:
        rows.append({
            "Filename":                 fname,
            "Label":                    int(rp.label),
            "Mean_intensity":           float(rp.intensity_mean),
            "Integrated_intensity":     float(rp.intensity_mean * rp.area),
            "Area_um2":                 float(rp.area * px_area),
            "Equivalent_diameter_um":   float(rp.equivalent_diameter_area * pixel_size_um),
            "Perimeter_um":             float(rp.perimeter * pixel_size_um),
            "Eccentricity":             float(rp.eccentricity),
            "Solidity":                 float(rp.solidity),   # NEW: compactness measure
            "Centroid_yx_px":           (float(rp.centroid[0]), float(rp.centroid[1]))
        })

    if plot:
        fig, ax = plt.subplots(1, 5, figsize=(25, 5))
        ax[0].imshow(plane, cmap="gray")
        ax[0].set_title("DAPI channel"); ax[0].axis("off")

        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title(f"Mask (Otsu×{otsu_scale:.2f})"); ax[1].axis("off")

        ax[2].imshow(dist_s, cmap="viridis")
        ax[2].set_title("Distance map (smoothed)"); ax[2].axis("off")

        seed_vis = np.zeros_like(plane, dtype=float)
        if len(coords) > 0:
            seed_vis[tuple(coords.T)] = 1.0
        ax[3].imshow(mask, cmap="gray")
        ax[3].imshow(seed_vis, alpha=0.9, cmap="hot")
        ax[3].set_title(f"h-maxima seeds: {len(coords)}"); ax[3].axis("off")

        ax[4].imshow(nuc_labels, cmap="nipy_spectral", interpolation="nearest")
        ax[4].set_title(f"Final nuclei (N={int(nuc_labels.max())})"); ax[4].axis("off")

        plt.suptitle(fname, fontsize=9)
        plt.tight_layout()
        plt.show()

    return nuc_labels, rows


# -------------- Actin segmentation (unchanged logic, minor cleanup) --------------
def segment_actin_from_nuclei(
    actin_plane, nuclei_labels,
    pixel_size_um, min_area_um2,
    otsu_scale_act, gauss_sigma_px,
    extra_planes=None,
    plot=False, fname=""
):
    a = actin_plane.astype(np.float32)
    p1, p99 = np.percentile(a, (1, 99.8))
    a = np.clip((a - p1) / (p99 - p1 + 1e-6), 0, 1)
    a_s = gaussian_filter(a, sigma=1.0)

    thr = filters.threshold_otsu(a_s) * otsu_scale_act
    mask = a_s > thr

    min_area_px = _min_area_to_px(min_area_um2, pixel_size_um)
    mask = remove_small_holes(mask, area_threshold=int(4 * min_area_px))
    mask = _binary_clean(mask, min_area_px)
    mask = binary_opening(mask, footprint=disk(2))

    nuc_seeds = _centroids_from_labels(nuclei_labels)
    markers = np.zeros_like(a_s, dtype=np.int32)
    for nuc_id, y, x in nuc_seeds:
        markers[y, x] = int(nuc_id)

    seed_dil = binary_dilation(markers > 0, footprint=disk(2))
    ws_mask = np.logical_or(mask, seed_dil)

    grad = filters.sobel(a_s)
    cost = 0.7 * grad + 0.3 * (1.0 - a_s)
    actin_labels = watershed(cost, markers=markers, mask=ws_mask, compactness=2.0)

    expected = int(nuclei_labels.max())
    if int(actin_labels.max()) < expected:
        dist = distance_transform_edt(ws_mask)
        actin_labels = watershed(-dist, markers=markers, mask=ws_mask, compactness=1.0)

    px_area = pixel_size_um ** 2
    props_a = regionprops(actin_labels, intensity_image=actin_plane)

    extra_means = {}
    if extra_planes:
        for prefix, plane in extra_planes.items():
            rp_extra = regionprops(actin_labels, intensity_image=plane.astype(np.float32))
            extra_means[prefix] = {rp.label: float(rp.intensity_mean) for rp in rp_extra}

    rows = []
    for rp in props_a:
        lab = int(rp.label)
        row = {
            "Filename":                         fname,
            "Label":                            lab,
            "Actin_mean_intensity":             float(rp.intensity_mean),
            "Actin_integrated_intensity":       float(rp.intensity_mean * rp.area),
            "Actin_area_um2":                   float(rp.area * px_area),
            "Actin_equivalent_diameter_um":     float(rp.equivalent_diameter_area * pixel_size_um),
            "Actin_perimeter_um":               float(rp.perimeter * pixel_size_um),
            "Actin_centroid_yx_px":             (float(rp.centroid[0]), float(rp.centroid[1])),
        }
        for prefix, m in extra_means.items():
            mean_val = m.get(lab, np.nan)
            row[f"{prefix}_mean_intensity"] = mean_val
            row[f"{prefix}_integrated_intensity"] = (float(mean_val * rp.area)
                                                      if np.isfinite(mean_val) else np.nan)
        rows.append(row)

    if plot:
        seed_vis = np.zeros_like(a_s, dtype=float)
        for _, y, x in nuc_seeds:
            seed_vis[y, x] = 1.0
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(actin_plane, cmap="gray"); ax[0].set_title("Actin channel"); ax[0].axis("off")
        ax[1].imshow(mask, cmap="gray"); ax[1].set_title(f"Actin mask (Otsu×{otsu_scale_act:.2f})"); ax[1].axis("off")
        ax[2].imshow(ws_mask, cmap="gray"); ax[2].imshow(seed_vis, alpha=0.9); ax[2].set_title("WS mask + seeds"); ax[2].axis("off")
        ax[3].imshow(actin_labels, cmap="nipy_spectral", interpolation="nearest")
        ax[3].set_title(f"Actin watershed (N={int(actin_labels.max())})"); ax[3].axis("off")
        plt.tight_layout(); plt.show()

    return actin_labels, rows


# ---------------- Batch runner ----------------
def process_images(image_dir,
                   pixel_size_um=PIXEL_SIZE_UM,
                   min_area_um2=MIN_AREA_UM2,
                   max_area_um2=MAX_AREA_UM2,
                   nuc_channel=NUCLEI_CHANNEL,
                   act_channel=ACTIN_CHANNEL,
                   myosin_channel=MYOSIN_CHANNEL,
                   pmyosin_channel=PMYOSIN_CHANNEL,
                   plot=PLOT_DEBUG):
    all_nuc_rows = []
    all_act_rows = []

    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith((".tif", ".tiff")):
            continue
        path = os.path.join(image_dir, filename)
        img = tiff.imread(path)

        nuc_plane  = _extract_channel_plane(img, nuc_channel)
        act_plane  = _extract_channel_plane(img, act_channel)
        myo_plane  = _extract_channel_plane(img, myosin_channel)
        pmyo_plane = _extract_channel_plane(img, pmyosin_channel)

        nuc_labels, nuc_rows = segment_nuclei(
            nuc_plane, pixel_size_um, min_area_um2, max_area_um2,
            OTSU_SCALE_NUC, GAUSS_SIGMA_PX, EXCLUDE_BORDER,
            PEAK_MIN_DIST_UM, H_MAXIMA_H, MAX_ECCENTRICITY,
            plot=plot, fname=filename
        )
        all_nuc_rows.extend(nuc_rows)

        act_labels, act_rows = segment_actin_from_nuclei(
            act_plane, nuc_labels,
            pixel_size_um, min_area_um2,
            OTSU_SCALE_ACT, GAUSS_SIGMA_PX,
            extra_planes={"Myosin": myo_plane, "pMyosin": pmyo_plane},
            plot=plot, fname=filename
        )
        all_act_rows.extend(act_rows)

        if SAVE_LABEL_TIFS:
            out_dir = OUTPUT_DIR or image_dir
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(filename)[0]
            tiff.imwrite(os.path.join(out_dir, f"{base}_labels_nuclei.tif"),
                         nuc_labels.astype(np.uint16))
            tiff.imwrite(os.path.join(out_dir, f"{base}_labels_actin.tif"),
                         act_labels.astype(np.uint16))

    return all_nuc_rows, all_act_rows


def results_to_dataframes(nuc_rows, act_rows):
    nuc_cols = ["Filename", "Label", "Mean_intensity", "Integrated_intensity",
                "Area_um2", "Equivalent_diameter_um", "Perimeter_um",
                "Eccentricity", "Solidity", "Centroid_yx_px"]
    act_cols = [
        "Filename", "Label",
        "Actin_mean_intensity", "Actin_integrated_intensity",
        "Actin_area_um2", "Actin_equivalent_diameter_um",
        "Actin_perimeter_um", "Actin_centroid_yx_px",
        "Myosin_mean_intensity", "Myosin_integrated_intensity",
        "pMyosin_mean_intensity", "pMyosin_integrated_intensity"
    ]
    return pd.DataFrame(nuc_rows, columns=nuc_cols), pd.DataFrame(act_rows, columns=act_cols)


# ---------------- Example usage ----------------
if __name__ == "__main__":
    image_directory = "/Volumes/giuliaam/LSM980/RC1.2"  # <- update this

    nuc_rows, act_rows = process_images(
        image_directory,
        pixel_size_um=PIXEL_SIZE_UM,
        min_area_um2=MIN_AREA_UM2,
        max_area_um2=MAX_AREA_UM2,
        nuc_channel=NUCLEI_CHANNEL,
        act_channel=ACTIN_CHANNEL,
        myosin_channel=MYOSIN_CHANNEL,
        pmyosin_channel=PMYOSIN_CHANNEL,
        plot=True
    )
    df_nuc, df_act = results_to_dataframes(nuc_rows, act_rows)

    df_all = (pd.merge(df_nuc, df_act, on=["Filename", "Label"], how="outer")
                .sort_values(["Filename", "Label"]))
    print(df_all.head())
    # df_all.to_csv("nuc_act_myo_pmyo_results.csv", index=False)
