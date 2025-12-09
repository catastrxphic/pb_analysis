import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import pathlib as Patch
import optparse as opt
from scipy.spatial import distance
import sys
import glob


# ensure yolov7 package is importable when running this script from the repo root or other CWDs
YOLOV7_DIR = Patch.Path(__file__).resolve().parent / "yolov7"
if str(YOLOV7_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV7_DIR))

import torch.serialization
from models.yolo import Model
torch.serialization.add_safe_globals([Model])

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_synchronized



# ______________ Config ___________
PIXEL_SIZE_UM = 1.0  # change when known
DATA_OUTPUT_DIR = "data"
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)


# _____________ Distance Analysis _____________
def get_centroid_from_bbox(xyxy):
    """ get centroid (x,y) from bounding box coordinates"""
    x1, y1, x2, y2 =xyxy
    cx = (x1+x2) / 2
    cy = (y1+y2) / 2
    return (cx,cy)

def compute_distances(pbody_centroids, target_centroids, pixel_size_um = 1.0):
    """ compute minimum distance from each pbody to nearest target object """
    # if there is no nuclei or mitochondrion, there will be "na" for each and all pbodies in image
    if not target_centroids:
        return[np.nan]*len(pbody_centroids)
    
    distances = []
    target_points = np.array(target_centroids)
    for centroid in pbody_centroids:
        dist = np.min(distance.cdist([centroid], target_points))
        distances.append(dist * pixel_size_um)
    return distances 

# _____________ YOLO detection analysis _____________
def analyze_with_yolo(weights, source, device='cpu', img_size=640, group_name="GX"):
    import glob  # ensure it's imported

    device = select_device(device)
    half = device.type != 'cpu'

    # --- Load model ---
    print(f"🔍 Loading model from: {weights}")
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)

    if half:
        model.half()

    # --- Recursively find all images in nested subfolders ---
    pattern = os.path.join(source, '**', '*.*')
    image_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo')
    image_files = [f for f in glob.glob(pattern, recursive=True) if f.lower().endswith(image_extensions)]

    if not image_files:
        raise FileNotFoundError(f"❌ No images found in {source}")

    # Use recursive pattern so YOLO can handle all subfolders
    dataset = LoadImages(os.path.join(source, '**', '*.*'), img_size=img_size, stride=stride)

    # --- Set up class colors & names ---
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [(255, 255, 0), (0, 255, 0), (0, 0, 255)]  # BGR for P-body, Nucleus, Mitochondria

    results = []
    save_dir = Patch.Path(DATA_OUTPUT_DIR) / f"{group_name}_edited"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Run inference ---
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)

        for det in pred:
            im0 = im0s.copy()
            base_name = os.path.basename(path)
            pbody_centroids, nucleus_centroids, mito_centroids = [], [], []

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    cls = int(cls)
                    centroid = get_centroid_from_bbox(xyxy)

                    if cls == 0:
                        pbody_centroids.append(centroid)
                    elif cls == 1:
                        nucleus_centroids.append(centroid)
                    elif cls == 2:
                        mito_centroids.append(centroid)

                    # Draw bounding boxes
                    label = f"{names[cls]} {conf:.2f}"
                    color = colors[cls]
                    cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])),
                                  (int(xyxy[2]), int(xyxy[3])), color, 1)
                    cv2.putText(im0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Compute distances
            dist_nucleus = compute_distances(pbody_centroids, nucleus_centroids, PIXEL_SIZE_UM)
            dist_mito = compute_distances(pbody_centroids, mito_centroids, PIXEL_SIZE_UM)

            # Save annotated images
            save_path = save_dir / base_name
            cv2.imwrite(str(save_path), im0)

            for i, (dn, dm) in enumerate(zip(dist_nucleus, dist_mito)):
                results.append((group_name, base_name, i + 1, dn, dm))

    # --- Save CSV ---
    csv_name = f"{group_name}_analysis_distance.csv"
    csv_path = os.path.join(DATA_OUTPUT_DIR, csv_name)
    df = pd.DataFrame(results, columns=["group", "image", "pbody_id",
                                        "distance_to_nucleus_um", "distance_to_mitochondria_um"])
    df.to_csv(csv_path, index=False)
    print(f"✅ Analysis complete for {group_name}. Results saved to {csv_path}")

    # --- Summarize results ---
    summarize_results(csv_path)
    print(f"🏁 Processing done for group {group_name} in {time.time() - t0:.2f} seconds.")



# ____________ Summarize ____________
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


# ____________ Main ____________
if __name__ == "__main__":
    weights = input("Enter path to YOLO weights file: ").strip()
    parent_folder = input("Enter path to parent folder containing groups: ").strip()
    device = input("Enter device (cpu or cuda): ").strip().lower() or 'cpu'

    for group_name in os.listdir(parent_folder):
        group_path = os.path.join(parent_folder, group_name)
        if not os.path.isdir(group_path):
            continue
        
        print(f"\n🧬 Processing group: {group_name}")
        analyze_with_yolo(weights, group_path, device=device, group_name=group_name)

    print("\nAll groups processed succesfully")

