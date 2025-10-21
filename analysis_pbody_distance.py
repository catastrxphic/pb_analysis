import cv2
import numpy as np
import pandas as pd
import os
from scipy.spatial import distance

# ---------- CONFIG ----------
PIXEL_SIZE_UM = 1.0  # leave 1.0 if unknown
DATA_OUTPUT_DIR = "data"  # all analysis CSVs will go here

# ---------- IMAGE ENHANCEMENT ----------
def enhance_image(img):
    """Enhance visibility of blue (nucleus) and purple (mitochondria) regions."""
    img_float = img.astype(np.float32) / 255.0
    img_float[..., 0] = np.clip(img_float[..., 0] * 2.5, 0, 1)  # Blue
    img_float[..., 2] = np.clip(img_float[..., 2] * 2.5, 0, 1)  # Red
    lab = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

# ---------- MASK DETECTION ----------
def detect_objects(img):
    """Extract binary masks for P-bodies (green), nuclei (blue), and mitochondria (purple)."""
    b, g, r = cv2.split(img)
    _, green_mask = cv2.threshold(g, 50, 255, cv2.THRESH_BINARY)
    _, blue_mask = cv2.threshold(b, 40, 255, cv2.THRESH_BINARY)
    _, red_mask = cv2.threshold(r, 40, 255, cv2.THRESH_BINARY)
    purple_mask = cv2.bitwise_and(red_mask, blue_mask)
    return green_mask, blue_mask, purple_mask

def get_centroids(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    return [tuple(map(float, c)) for c in centroids[1:]]

def compute_distances(pbody_centroids, target_mask, pixel_size_um=1.0):
    target_points = np.column_stack(np.where(target_mask > 0))
    if target_points.size == 0:
        return [np.nan] * len(pbody_centroids)
    distances = []
    for p in pbody_centroids:
        d = np.min(distance.cdist([p[::-1]], target_points))
        distances.append(d * pixel_size_um)
    return distances

# ---------- IMAGE ANALYSIS ----------
def analyze_image(image_path, save_dir, group_name):
    img = cv2.imread(image_path)
    enhanced = enhance_image(img)
    green_mask, blue_mask, purple_mask = detect_objects(enhanced)
    pbody_centroids = get_centroids(green_mask)

    dist_nucleus = compute_distances(pbody_centroids, blue_mask, PIXEL_SIZE_UM)
    dist_mito = compute_distances(pbody_centroids, purple_mask, PIXEL_SIZE_UM)

    # Save enhanced image to edited folder
    base_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, base_name)
    cv2.imwrite(save_path, enhanced)

    return [
        (group_name, os.path.basename(image_path), i + 1, dn, dm)
        for i, (dn, dm) in enumerate(zip(dist_nucleus, dist_mito))
    ]

# ---------- SUMMARIZE RESULTS ----------
def summarize_results(csv_path):
    df = pd.read_csv(csv_path)
    summary = (
        df.groupby(["group", "image"])
          .agg(
              mean_dist_nucleus=("distance_to_nucleus_um", "mean"),
              mean_dist_mito=("distance_to_mitochondria_um", "mean"),
              n_pbody=("pbody_id", "count")
          )
          .reset_index()
    )
    summary_path = csv_path.replace("_analysis_distance.csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"✅ Summary saved to {summary_path}")

# ---------- MAIN PIPELINE ----------
def process_parent_folder(parent_folder):
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    for group_name in os.listdir(parent_folder):  # G1–G5
        group_path = os.path.join(parent_folder, group_name)
        if not os.path.isdir(group_path):
            continue

        print(f"\n🧬 Processing group: {group_name}")
        all_results = []

        for medicine_name in os.listdir(group_path):
            medicine_path = os.path.join(group_path, medicine_name)
            if not os.path.isdir(medicine_path):
                continue

            # Create edited image folder
            edited_folder_name = f"{group_name}_{medicine_name}_editted"
            edited_folder_path = os.path.join(medicine_path, edited_folder_name)
            os.makedirs(edited_folder_path, exist_ok=True)

            for filename in os.listdir(medicine_path):
                if filename.lower().endswith(".png"):
                    img_path = os.path.join(medicine_path, filename)
                    print(f"   → {filename}")
                    results = analyze_image(img_path, edited_folder_path, group_name)
                    all_results.extend(results)

            # Save one CSV per medicine folder
            csv_name = f"{group_name}_{medicine_name}_analysis_distance.csv"
            csv_path = os.path.join(DATA_OUTPUT_DIR, csv_name)
            df = pd.DataFrame(
                all_results,
                columns=["group", "image", "pbody_id", "distance_to_nucleus_um", "distance_to_mitochondria_um"]
            )
            df.to_csv(csv_path, index=False)
            print(f"📊 Saved data to {csv_path}")

            # Summarize
            summarize_results(csv_path)

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    parent_folder = input("Enter the path to the parent folder (e.g., test or train): ").strip()
    process_parent_folder(parent_folder)
    print("\n🎉 Processing complete for all groups and medicines!")
