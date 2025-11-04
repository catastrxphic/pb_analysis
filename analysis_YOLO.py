import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import pathlib as Patch
import optparse as opt
from scipy.spatial import distance
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_mas_suppression, scale_coords
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
def analyze_with_yolo(weights, source, device='cpu', img_size= opt.source, group_name= "GX"):
    device = select_device(device)
    half = device.type  != 'cpu'
    model = attempt_load(weights, map_location = device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)

    if half:
        model.half

    dataset = LoadImages(source, img_size = img_size, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [(255, 255, 0), (0, 255, 0), (0, 0, 255)]  # BGR for P-body, Nucleus, Mitochondria

    results = []
    save_dir = Patch.Path(DATA_OUTPUT_DIR) / f"{group_name}_edited"
    save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for path, img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # inference
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

                    if cls == 0:  # P-body
                        pbody_centroids.append(centroid)
                    elif cls == 1:  # Nucleus
                        nucleus_centroids.append(centroid)
                    elif cls == 2:  # Mitochondria
                        mito_centroids.append(centroid)

                    # draw bounding boxes
                    label = f"{names[cls]} {conf:.2f}"
                    color = colors[cls]
                    cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 1)
                    cv2.putText(im0, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # compute distances
            dist_nucleus = compute_distances(pbody_centroids, nucleus_centroids, PIXEL_SIZE_UM)
            dist_mito = compute_distances(pbody_centroids, mito_centroids, PIXEL_SIZE_UM)

            # save annotated images
            save_path = save_dir / base_name
            cv2.imwrite(str(save_path), im0)

            for i, (dn, dm) in enumerate(zip(dist_nucleus, dist_mito)):
                results.append((group_name, base_name, i + 1, dn, dm))

    # save CSV
    csv_name = f"{group_name}_analysis_distance.csv"
    csv_path = os.path.join(DATA_OUTPUT_DIR, csv_name)
    df = pd.DataFrame(results, columns=["group", "image", "pbody_id", "distance_to_nucleus_um", "distance_to_mitochondria_um"])
    df.to_csv(csv_path, index=False)
    print(f"✅ Analysis complete for {group_name}. Results saved to {csv_path}")

    # summary 
    summarize_results(csv_path)
    print( "Processing done for group {group_name} in {time.time() - t0:.2f} seconds.")


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

    print("\nAll groups processed succesfully.")

