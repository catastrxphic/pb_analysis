# **PB Analysis - PB-scope-detection Continuation**

## **Introduction**

This is a Python-based image-analysis pipeline designed to extract geometric features from microscopy images, including:

* **Center-to-center distances** between fluorescent puncta,
* **Nearest-neighbor relationships**, and
* **Statistical summaries** of spatial patterns.

This workflow consists of three components:

1. **Preprocessing** (ImageJ / thresholding / segmentation),
2. **Feature extraction** (Python), and
3. **Distance computation & visualization** (Python).

Before starting, ensure you have the following dependencies:

* **Python 3.8+**
* **OpenCV**
* **NumPy**
* **SciPy**
* **Pillow (PIL)**
* **Matplotlib**
* **scikit-image**

---

## **Framework of the Distance-Measurement Pipeline**

The workflow operates in three stages:

1. **Image Preprocessing**
   Users can apply thresholding, segmentation, or filtering in ImageJ or Cellpose.
   Output files are passed to the detection script.

2. **Object Detection & Coordinate Extraction**
   The Python script:

   * Loads the image,
   * Finds all particles/structures,
   * Computes centroids, and bounding contours.

3. **Distance & Radius Analysis**
   For each detected structure:

   * Computes Euclidean distances,
   * Determines nearest neighbors,
   * Generates summary statistics and plots.

This workflow enables reproducible quantification of p-body spacing.

---

### **Image Processing**

To compute distances, radii, and nearest-neighbor relationships for a single image use pbodybw.py

This script will:

1. Load the image
2. Apply grayscale conversion
3. Detect contours / particles
4. Compute:

   * Centroids (x, y)
   * Radius estimation
   * Euclidean distances
   * Nearest neighbors
5. Generate visual overlays
6. Save:

   * A CSV file of coordinates and distances

It also performs the full extraction pipeline for all images and outputs:

* A merged CSV of all detected distances
* Per-image diagnostic plots
* A global statistical summary


---

## **How the Script Works (Step-by-Step)**

The main detection script proceeds as follows:

1. **Load Image**
   Converts to grayscale and optionally resizes.

2. **Denoising & Thresholding**
   Applies Gaussian blur and adaptive thresholding to isolate particles.

3. **Contour/Feature Detection**

   * Uses OpenCV to find closed contours
   * Computes each particle’s centroid
   * Estimates radius via contour area

4. **Distance Computation**

   * Computes all pairwise distances
   * Identifies nearest neighbors
   * Removes zero-distance duplicates

5. **Output Generation**

   * Annotated visualization (circles + indexed labels)
   * CSV file with:

     * ID, x, y, radius
     * Distance to every other particle
     * Nearest-neighbor distance


---

## **Citation**

If you use this repository, please cite the associated methodology:

```
@article {Shen2025.06.14.659731,
	author = {Shen, Dexin and Zhu, Qionghua and Pang, Xiquan and Yang, Xian and Pan, Dongzhen and Zhang, Mengyang and Li, Yanping and Sun, Zhiyuan and Fang, Liang and Chen, Wei and Tsuboi, Tatsuhisa},
	title = {PB-scope: Contrastive learning of dynamic processing body formation reveals undefined mechanisms of approved compounds},
	elocation-id = {2025.06.14.659731},
	year = {2025},
	doi = {10.1101/2025.06.14.659731},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/06/15/2025.06.14.659731},
	eprint = {https://www.biorxiv.org/content/early/2025/06/15/2025.06.14.659731.full.pdf},
	journal = {bioRxiv}
}
```

---

## **License**

This project is licensed under the GNU License.
See `LICENSE` for details.


