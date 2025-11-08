#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
PIXEL_SIZE_UM = 1.0       # set to real microns per pixel if known
RESULTS_ROOT = "results"  # will contain per-group subfolders with per-image Excel files
DATA_OUTPUT_DIR = "data"  # optional top-level CSVs per medicine (kept for convenience)

# ---------- UTILITIES ----------
def centroid_from_contour(contour):
    M = cv2.moments(contour)
    if M.get("m00", 0) == 0:
        pts = contour.reshape(-1, 2)
        return tuple(np.mean(pts, axis=0))
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])

def min_distances(points, targets):
    """Return numpy array of min distance from each point to closest target.
       points: (N,2) array-like; targets: (M,2) array-like.
       If targets empty -> returns array of np.nan of length N."""
    pts = np.array(points)
    if pts.size == 0:
        return np.array([], dtype=float)
    t = np.array(targets)
    if t.size == 0:
        return np.full((len(pts),), np.nan)
    # compute pairwise distances efficiently
    d2 = ((pts[:, None, :] - t[None, :, :]) ** 2).sum(axis=2)
    mins = np.sqrt(d2.min(axis=1))
    return mins

# ---------- DETECTION FUNCTIONS ----------
def detect_pbodies_from_binary(gray):
    """
    pbodies: expects a black/white layer in the image (grayscale).
    Uses Otsu to handle variable brightness automatically.
    Returns list of centroids and contours.
    """
    # Otsu thresholding to get binary pbody mask
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # optional: remove small noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    kept_contours = []
    for c in contours:
        if cv2.contourArea(c) < 6:  # filter tiny areas (tune if needed)
            continue
        kept_contours.append(c)
        centroids.append(centroid_from_contour(c))
    return centroids, kept_contours, mask

def detect_nuclei_from_blue(img_bgr):
    """Detect nuclei using blue channel (BGR ordering). Returns centroids, contours, mask."""
    blue = img_bgr[:, :, 0]
    # Otsu threshold on blue channel
    _, mask = cv2.threshold(blue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids, kept = [], []
    for c in contours:
        if cv2.contourArea(c) < 8:
            continue
        kept.append(c)
        centroids.append(centroid_from_contour(c))
    return centroids, kept, mask

def detect_mito_from_purple(img_bgr):
    """
    Approximate purple by combining red and blue channels.
    You can tune the blending weights if your purple is more red-leaning.
    """
    red = img_bgr[:, :, 2].astype(np.float32)
    blue = img_bgr[:, :, 0].astype(np.float32)
    purple = cv2.normalize((0.5 * red + 0.5 * blue).astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(purple, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids, kept = [], []
    for c in contours:
        if cv2.contourArea(c) < 8:
            continue
        kept.append(c)
        centroids.append(centroid_from_contour(c))
    return centroids, kept, mask

# ---------- IMAGE ANALYSIS ----------
def analyze_single_image(image_path, edited_folder, group_name, medicine_name, draw=True):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect pbodies (binary layer)
    p_centroids, p_contours, p_mask = detect_pbodies_from_binary(gray)

    # detect nuclei (blue) and mitochondria (purple)
    n_centroids, n_contours, n_mask = detect_nuclei_from_blue(img)
    m_centroids, m_contours, m_mask = detect_mito_from_purple(img)

    # distances (pixels) -> multiply by PIXEL_SIZE_UM if you want micrometers
    d_to_n = min_distances(p_centroids, n_centroids) * PIXEL_SIZE_UM if len(p_centroids) > 0 else np.array([])
    d_to_m = min_distances(p_centroids, m_centroids) * PIXEL_SIZE_UM if len(p_centroids) > 0 else np.array([])

    # per-image stds (NaN if no pbodies or no targets)
    nuclei_std = np.nan if d_to_n.size == 0 else np.nanstd(d_to_n)
    mito_std = np.nan if d_to_m.size == 0 else np.nanstd(d_to_m)

    # draw annotated image for verification
    if draw:
        vis = img.copy()
        for c in p_contours:
            cv2.drawContours(vis, [c], -1, (0, 255, 0), 1)
        for c in n_contours:
            cv2.drawContours(vis, [c], -1, (255, 0, 0), 1)
        for c in m_contours:
            cv2.drawContours(vis, [c], -1, (255, 0, 255), 1)
        for (cx, cy) in p_centroids:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 3, (0, 255, 255), -1)
        # save annotated
        os.makedirs(edited_folder, exist_ok=True)
        base = os.path.basename(image_path)
        cv2.imwrite(os.path.join(edited_folder, base), vis)

    # prepare rows for each pbody
    rows = []
    for idx, (cxcy) in enumerate(p_centroids, start=1):
        dn = float(d_to_n[idx - 1]) if d_to_n.size > 0 else np.nan
        dm = float(d_to_m[idx - 1]) if d_to_m.size > 0 else np.nan
        rows.append({
            "image": os.path.basename(image_path),
            "pbody_number": idx,
            "dist_nuclei": dn,
            "dist_mito": dm,
            "nuclei_std": float(nuclei_std) if not np.isnan(nuclei_std) else np.nan,
            "mito_std": float(mito_std) if not np.isnan(mito_std) else np.nan,
            "medicine": medicine_name,
            "group": ("DMSO" if group_name == "G0" else group_name)
        })

    # if there are zero pbodies, we still write an empty-data row? we will return empty list (no rows)
    return rows, len(p_centroids), len(n_centroids), len(m_centroids)

# ---------- FOLDER WALK & SAVE ----------
def process_parent_folder(parent_folder):
    parent_folder = os.path.abspath(parent_folder)
    results_root = os.path.join(os.path.dirname(parent_folder), RESULTS_ROOT)
    data_output_dir = os.path.join(os.path.dirname(parent_folder), DATA_OUTPUT_DIR)
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)

    for group_name in sorted(os.listdir(parent_folder)):
        gpath = os.path.join(parent_folder, group_name)
        if not os.path.isdir(gpath):
            continue
        print(f"\nProcessing group: {group_name}")

        group_results_folder = os.path.join(results_root, group_name)
        os.makedirs(group_results_folder, exist_ok=True)

        for medicine_name in sorted(os.listdir(gpath)):
            mpath = os.path.join(gpath, medicine_name)
            if not os.path.isdir(mpath):
                continue
            print(f"  Medicine: {medicine_name}")

            edited_folder = os.path.join(mpath, f"{group_name}_{medicine_name}_editted")
            os.makedirs(edited_folder, exist_ok=True)

            # will collect per-medicine rows (optional saving CSV)
            medicine_rows = []

            # iterate files in medicine folder
            for fname in sorted(os.listdir(mpath)):
                fpath = os.path.join(mpath, fname)
                if not os.path.isfile(fpath):
                    continue
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    continue

                try:
                    rows, n_pb, n_nuc, n_mito = analyze_single_image(fpath, edited_folder, group_name, medicine_name, draw=True)
                except Exception as e:
                    print(f"    ! Failed to process {fname}: {e}")
                    continue

                # save one Excel per image into results/<GROUP>/
                image_base = os.path.splitext(fname)[0]
                image_out_file = os.path.join(group_results_folder, f"{group_name}_{medicine_name}_{image_base}_analysis.xlsx")
                if rows:
                    pd.DataFrame(rows).to_excel(image_out_file, index=False)
                else:
                    # write empty dataframe with headers (so you have a file even if no pbodies)
                    cols = ["image", "pbody_number", "dist_nuclei", "dist_mito", "nuclei_std", "mito_std", "medicine", "group"]
                    pd.DataFrame(columns=cols).to_excel(image_out_file, index=False)

                # accumulate per-medicine (optional)
                medicine_rows.extend(rows)

                print(f"    processed {fname}: pbodies={n_pb}, nuclei={n_nuc}, mito={n_mito}")

            # optional: write per-medicine CSV to data_output_dir
            med_csv = os.path.join(data_output_dir, f"{group_name}_{medicine_name}_analysis_distance.csv")
            if medicine_rows:
                pd.DataFrame(medicine_rows).to_csv(med_csv, index=False)
            else:
                # empty CSV with columns
                cols = ["image", "pbody_number", "dist_nuclei", "dist_mito", "nuclei_std", "mito_std", "medicine", "group"]
                pd.DataFrame(columns=cols).to_csv(med_csv, index=False)

    print("\nAll processing finished.")

# ---------- ENTRY ----------
if __name__ == "__main__":
    parent = input("Enter path to parent folder (e.g. dataset_t6_grouped/test): ").strip()
    process_parent_folder(parent)
