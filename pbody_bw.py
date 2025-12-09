# Imports needed to run the code
import os
import cv2
import numpy as np
import pandas as pd

# ---------- CONFIG ----------
PIXEL_SIZE_UM = 0.1625       # conversion from micrometer (depends on microscope) to pixel
RESULTS_ROOT = "results2"    # where the results are to be stored
DATA_OUTPUT_DIR = "data2"    # top-level CSVs / summary Excels per medicine & global

# ---------- see the center of a point from a contour or conglomeration ----------
def centroid_from_contour(contour):
    M = cv2.moments(contour)
    if M.get("m00", 0) == 0:
        pts = contour.reshape(-1, 2)
        return tuple(np.mean(pts, axis=0))
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])

# ---------- calculate minimum distance from one point to another ----------
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

# ---------- draw distance lines - for double-checking algorithm performance ----------
def draw_distance_lines(img, p_centroids, n_centroids, m_centroids, d_to_n, d_to_m, save_path):
    """
    Draws one line from each p-body to its nearest nucleus (blue) and one line from each p-body 
    to its nearest mitochondrion (purple), labels distances in micrometers,
    they are saved and the combined image to stored in save_path variable
    """
    vis = img.copy()
    h, w = vis.shape[:2]

    p = np.array(p_centroids) if len(p_centroids) > 0 else np.empty((0,2))
    n = np.array(n_centroids) if len(n_centroids) > 0 else np.empty((0,2))
    m = np.array(m_centroids) if len(m_centroids) > 0 else np.empty((0,2))

    for i, (px, py) in enumerate(p):
        px_i, py_i = int(round(px)), int(round(py))

        # Nearest nucleus (BLUE)
        if n.size > 0:
            # compute distances to all nuclei and pick nearest
            dists_n = np.sqrt(((n - np.array([px, py])) ** 2).sum(axis=1))
            nn_idx = int(np.argmin(dists_n))
            nx, ny = int(round(n[nn_idx, 0])), int(round(n[nn_idx, 1]))
            # draw line
            cv2.line(vis, (px_i, py_i), (nx, ny), (255, 0, 0), 1)
            # label distance if available
            if d_to_n.size > i and not np.isnan(d_to_n[i]):
                label = f"{d_to_n[i]:.2f} μm"
                midx, midy = (px_i + nx) // 2, (py_i + ny) // 2
                # add small background rectangle for readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                bx1, by1 = max(0, midx - 2), max(0, midy - th - 2)
                bx2, by2 = min(w-1, midx + tw + 2), min(h-1, midy + 2)
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0,0,0), -1)
                cv2.putText(vis, label, (midx, midy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Nearest mitochondrion (PURPLE)
        if m.size > 0:
            dists_m = np.sqrt(((m - np.array([px, py])) ** 2).sum(axis=1))
            mm_idx = int(np.argmin(dists_m))
            mx, my = int(round(m[mm_idx, 0])), int(round(m[mm_idx, 1]))
            # draw line (purple)
            cv2.line(vis, (px_i, py_i), (mx, my), (255, 0, 255), 1)
            if d_to_m.size > i and not np.isnan(d_to_m[i]):
                label = f"{d_to_m[i]:.2f} μm"
                midx, midy = (px_i + mx) // 2, (py_i + my) // 2
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                bx1, by1 = max(0, midx - 2), max(0, midy - th - 2)
                bx2, by2 = min(w-1, midx + tw + 2), min(h-1, midy + 2)
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0,0,0), -1)
                cv2.putText(vis, label, (midx, midy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    # Save combined image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)

# ---------- detect the pbodies from the binary mask ----------
def detect_pbodies_from_binary(gray):
    """
    pbodies are detected by using grayscale binary mask with adaptive cleaning.
    function return the centroids, kept_contours, and mask.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # invert mask (ensure pbodies = white)
    if np.mean(mask[mask > 0]) < 128:
        mask = cv2.bitwise_not(mask)

    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids, kept_contours = [], []
    for c in contours:
        area = cv2.contourArea(c)
        # Filtering blobs by area 
        if area < 0 or area > 2000:
            continue
        kept_contours.append(c)
        centroids.append(centroid_from_contour(c))

    return centroids, kept_contours, mask

# ---------- detect the nuclei from the blue channel ----------
def detect_nuclei_from_blue(img_bgr):
    """Detect nuclei using blue channel (BGR ordering). function returns centroids, contours, mask."""
    blue = img_bgr[:, :, 0]
    _, mask = cv2.threshold(blue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids, kept = [], []
    for c in contours:
        if cv2.contourArea(c) < 3:
            continue
        kept.append(c)
        centroids.append(centroid_from_contour(c))
    return centroids, kept, mask

# ---------- detect the mitochondria from purple estimate (purple= red + blue) ---------- 
def detect_mito_from_purple(img_bgr):
    """
    Approximate purple by combining red and blue channels.
    functions returns centroids, contours, mask.
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
        if cv2.contourArea(c) < 3:
            continue
        kept.append(c)
        centroids.append(centroid_from_contour(c))
    return centroids, kept, mask

# ---------- integration of functions to analyze the whole image ----------
def analyze_single_image(image_path, edited_folder, group_name, medicine_name, draw=True):
    """
    Each image analysis returns:
      rows (per-pbody detail list),
      n_pb, n_nuc, n_mito (counts),
      mean_dn, std_dn, mean_dm, std_dm (per-image summary in micrometers)
    Also stores:
      - annotated image (contours) into edited_folder
      - distance-lines image into edited_folder (<base>_distance_lines.png)
      - per-image distance summary Excel into edited_folder
      - per-image detailed Excel (existing behavior) saved by caller
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image {image_path}")

    # Use green channel for p-bodies as before
    gray = img[:, :, 1]

    # detect pbodies (binary layer)
    p_centroids, p_contours, p_mask = detect_pbodies_from_binary(gray)

    # detect nuclei (blue) and mitochondria (purple)
    n_centroids, n_contours, n_mask = detect_nuclei_from_blue(img)
    m_centroids, m_contours, m_mask = detect_mito_from_purple(img)

    # distances (pixels) -> convert to micrometers
    d_to_n_pixels = min_distances(p_centroids, n_centroids) if len(p_centroids) > 0 else np.array([])
    d_to_m_pixels = min_distances(p_centroids, m_centroids) if len(p_centroids) > 0 else np.array([])

    d_to_n = d_to_n_pixels * PIXEL_SIZE_UM if d_to_n_pixels.size > 0 else np.array([])
    d_to_m = d_to_m_pixels * PIXEL_SIZE_UM if d_to_m_pixels.size > 0 else np.array([])

    # per-image stds (NaN if no pbodies or no targets)
    nuclei_std = np.nan if d_to_n.size == 0 else np.nanstd(d_to_n)
    mito_std = np.nan if d_to_m.size == 0 else np.nanstd(d_to_m)

    # draw annotated image for verification (existing contour visualization)
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
        annotated_path = os.path.join(edited_folder, base)
        cv2.imwrite(annotated_path, vis)

    # -------- save the distance visualization  ----------
    base = os.path.basename(image_path)
    distance_img_path = os.path.join(
        edited_folder,
        f"{os.path.splitext(base)[0]}_distance_lines.png"
    )
    # draw lines using micrometer distances (d_to_n and d_to_m are already in µm)
    draw_distance_lines(
        img,
        p_centroids,
        n_centroids,
        m_centroids,
        d_to_n,
        d_to_m,
        distance_img_path
    )

    # prepare rows for each pbody (existing structure)
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

    # --------- save for teh summary excel, adds distance and std per image ----------
    summary_data = {
        "metric": [
            "mean_dist_nuclei",
            "std_dist_nuclei",
            "mean_dist_mito",
            "std_dist_mito"
        ],
        "value": [
            float(np.nanmean(d_to_n)) if d_to_n.size > 0 else np.nan,
            float(np.nanstd(d_to_n)) if d_to_n.size > 0 else np.nan,
            float(np.nanmean(d_to_m)) if d_to_m.size > 0 else np.nan,
            float(np.nanstd(d_to_m)) if d_to_m.size > 0 else np.nan
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_out_path = os.path.join(
        edited_folder,
        f"{os.path.splitext(os.path.basename(image_path))[0]}_distance_summary.xlsx"
    )
    summary_df.to_excel(summary_out_path, index=False)

    # return details & per-image numeric summary (means/stds in µm)
    mean_dn = float(np.nanmean(d_to_n)) if d_to_n.size > 0 else np.nan
    std_dn = float(np.nanstd(d_to_n)) if d_to_n.size > 0 else np.nan
    mean_dm = float(np.nanmean(d_to_m)) if d_to_m.size > 0 else np.nan
    std_dm = float(np.nanstd(d_to_m)) if d_to_m.size > 0 else np.nan

    return rows, len(p_centroids), len(n_centroids), len(m_centroids), mean_dn, std_dn, mean_dm, std_dm

# ---------- function to get to folder and save data ----------
def process_parent_folder(parent_folder):
    parent_folder = os.path.abspath(parent_folder)
    results_root = os.path.join(os.path.dirname(parent_folder), RESULTS_ROOT)
    data_output_dir = os.path.join(os.path.dirname(parent_folder), DATA_OUTPUT_DIR)
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)

    # global accumulator for all images across all groups/medicines
    all_summary_rows = []

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

            # will collect per-medicine rows (original per-pbody rows)
            medicine_rows = []
            # collect per-medicine image-level summaries (mean/std)
            med_summary_rows = []

            # iterate files in medicine folder
            for fname in sorted(os.listdir(mpath)):
                fpath = os.path.join(mpath, fname)
                if not os.path.isfile(fpath):
                    continue
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    continue

                try:
                    rows, n_pb, n_nuc, n_mito, mean_dn, std_dn, mean_dm, std_dm = analyze_single_image(
                        fpath, edited_folder, group_name, medicine_name, draw=True
                    )
                except Exception as e:
                    print(f"    ! Failed to process {fname}: {e}")
                    continue

                # save one Excel per image into results/<GROUP>/ (existing behavior)
                image_base = os.path.splitext(fname)[0]
                image_out_file = os.path.join(group_results_folder, f"{group_name}_{medicine_name}_{image_base}_analysis.xlsx")
                if rows:
                    pd.DataFrame(rows).to_excel(image_out_file, index=False)
                else:
                    cols = ["image", "pbody_number", "dist_nuclei", "dist_mito", "nuclei_std", "mito_std", "medicine", "group"]
                    pd.DataFrame(columns=cols).to_excel(image_out_file, index=False)

                # accumulate per-medicine pbody rows (existing)
                medicine_rows.extend(rows)

                # accumulate per-medicine summary row for this image (image-level summary)
                img_summary = {
                    "group": ("DMSO" if group_name == "G0" else group_name),
                    "medicine": medicine_name,
                    "image": fname,
                    "mean_dist_nuclei": mean_dn,
                    "std_dist_nuclei": std_dn,
                    "mean_dist_mito": mean_dm,
                    "std_dist_mito": std_dm
                }
                med_summary_rows.append(img_summary)
                all_summary_rows.append(img_summary)

                print(f"    processed {fname}: pbodies={n_pb}, nuclei={n_nuc}, mito={n_mito}")

            # temptative: write per-medicine CSV to data_output_dir (if exists)
            med_csv = os.path.join(data_output_dir, f"{group_name}_{medicine_name}_analysis_distance.csv")
            if medicine_rows:
                pd.DataFrame(medicine_rows).to_csv(med_csv, index=False)
            else:
                cols = ["image", "pbody_number", "dist_nuclei", "dist_mito", "nuclei_std", "mito_std", "medicine", "group"]
                pd.DataFrame(columns=cols).to_csv(med_csv, index=False)

            # -------- save in per=medicine summary file --------
            med_summary_xlsx = os.path.join(data_output_dir, f"{group_name}_{medicine_name}_summary.xlsx")
            if med_summary_rows:
                pd.DataFrame(med_summary_rows).to_excel(med_summary_xlsx, index=False)
            else:
                cols = ["group", "medicine", "image",
                        "mean_dist_nuclei", "std_dist_nuclei",
                        "mean_dist_mito", "std_dist_mito"]
                pd.DataFrame(columns=cols).to_excel(med_summary_xlsx, index=False)

    # -------- save overall sumary of all images --------
    global_summary_xlsx = os.path.join(data_output_dir, "all_images_summary.xlsx")
    if all_summary_rows:
        pd.DataFrame(all_summary_rows).to_excel(global_summary_xlsx, index=False)
    else:
        cols = ["group", "medicine", "image",
                "mean_dist_nuclei", "std_dist_nuclei",
                "mean_dist_mito", "std_dist_mito"]
        pd.DataFrame(columns=cols).to_excel(global_summary_xlsx, index=False)

    print(f"\nGlobal summary saved to: {global_summary_xlsx}")
    print("\nAll processing finished.")

# ---------- ENTRY ----------
if __name__ == "__main__":
    parent = input("Enter path to parent folder (e.g. dataset_t6_grouped/test): ").strip()
    process_parent_folder(parent)
