# Foreground Mask Extraction In Row Crop Images

This repository provides a simple pipeline to extract the **foreground mask** from **row crop RGB** images using depth estimation and histogram-based thresholding.

---

## 🔹 Pipeline Overview

1. **Clone our repository [Foreground-Mask-Extraction-In-Row-Crop-Images.git](https://github.com/djaliloh/Foreground-Mask-Extraction-In-Row-Crop-Images.git)**
    * Steps:
        1. clone :  git clone https://github.com/djaliloh/Foreground-Mask-Extraction-In-Row-Crop-Images.git
        2. cd Foreground-Mask-Extraction-In-Row-Crop-Images

2. **Depth Estimation**

   * We use [Depth-Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) to generate depth maps from RGB images.
   * Steps:

     1. Clone the official repository.
     2. Download the pre-trained weights (we use the `vitl` model).

3. **Foreground Mask Extraction**

   * From the generated depth maps, we apply the following steps:

     1. **Histogram Computation** – compute the depth histogram for each map.
     2. **Valley Detection** – identify the minima (valleys) between significant modes in the histogram.
     3. **Thresholding** – select the optimal threshold based on the valley position and binarize the depth map.
     4. **Mask Generation** – the resulting binary mask separates the **foreground** from the **background**.

---

## 🔹 Summary of What the Code Does

* Takes an RGB image.
* Estimates its depth map using Depth-Anything V2.
* Analyzes the histogram of the depth map to detect valleys (local minima).
* Selects the best threshold based on these minima.
* Generates a clean **foreground mask**.
---

<!-- ## 👤 Authors

- **Abdoul Djalil OUSSEINI H.** – Initial work & implementation 
- **Herearii MOUTERA** – Contribute to the initial work 
- **David ROUSSEAU** – Contribute to the initial work  -->


---
