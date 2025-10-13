import cv2
import numpy as np
import os
import pandas as pd
from scipy.spatial import distance

# --- Parameters ---
INPUT_FOLDER = "dataset_t6_grouped/test/G1/T0374"   # Folder containing .png images
OUTPUT_CSV = "pbody_distances.csv"
PIXEL_SIZE_UM = 1.0       # Set to 1 if you don't know microscope scale

# ------------------------------------------------------------
# Helper: Enhance faint color channels (makes nucleus & mito visible)
# ------------------------------------------------------------
def enhance_channel(channel):
    p2, p98 = np.percentile(channel, (2, 98))
    channel = np.clip(channel, p2, p98)
    channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
    return channel.astype(np.uint8)

# ------------------------------------------------------------
# Main image analysis function
# ------------------------------------------------------------
def analyze_image(path):
    # 1️⃣ Load image
    image = cv2.imread(path)
    if image is None:
        print(f"⚠️ Skipping {path}, could not read file.")
        return []

    # 2️⃣ Enhance channels (to reveal faint blue/purple)
    b, g, r = cv2.split(image)
    b = enhance_channel(b)
    g = enhance_channel(g)
    r = enhance_channel(r)
    image = cv2.merge([b, g, r])

    # Optional: save enhanced preview for inspection
    cv2.imwrite(f"enhanced_preview_{os.path.basename(path)}", image)

    # 3️⃣ Create color masks (threshold tuned to detect color ranges)
    # Green = P-bodies
    green_mask = cv2.inRange(g, 60, 255)

    # Blue = Nucleus
    blue_mask = cv2.inRange(b, 40, 255)

    # Purple = Mitochondria (combination of red and blue channels)
    purple_mask = cv2.inRange(r - b, -20, 50) & cv2.inRange(r, 40, 255) & cv2.inRange(b, 40, 255)

    # 4️⃣ Find connected components for P-bodies
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(green_mask)
    pbody_centroids = [tuple(map(float, c)) for c in centroids[1:]]  # skip background

    # 5️⃣ Compute distances
    dist_nucleus = compute_distances(pbody_centroids, blue_mask, PIXEL_SIZE_UM)
    dist_mito = compute_distances(pbody_centroids, purple_mask, PIXEL_SIZE_UM)

    # 6️⃣ Combine results
    results = []
    for i, (d_n, d_m) in enumerate(zip(dist_nucleus, dist_mito)):
        results.append({
            "image": os.path.basename(path),
            "pbody_id": i + 1,
            "distance_to_nucleus_um": d_n,
            "distance_to_mitochondria_um": d_m
        })
    return results

# ------------------------------------------------------------
# Helper: Compute distance from each p-body centroid to nearest pixel in target mask
# ------------------------------------------------------------
def compute_distances(pbody_centroids, mask, pixel_size_um):
    yx = np.column_stack(np.where(mask > 0))
    distances = []

    if len(yx) == 0:
        # No target detected (empty nucleus/mito)
        return [np.nan] * len(pbody_centroids)

    for p in pbody_centroids:
        p_yx = np.array([[p[1], p[0]]])
        d = np.min(distance.cdist(p_yx, yx))
        distances.append(d * pixel_size_um)
    return distances

# ------------------------------------------------------------
# Main driver: process all images in folder
# ------------------------------------------------------------
def main():
    all_results = []
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".png")]

    for filename in image_files:
        path = os.path.join(INPUT_FOLDER, filename)
        print(f"Processing: {filename}")
        results = analyze_image(path)
        all_results.extend(results)

    # Save CSV output
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Analysis complete! Results saved to {OUTPUT_CSV}")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
