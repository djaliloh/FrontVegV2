# Foreground Mask Extraction In Row Crop Images

This repository provides a simple pipeline to extract the **foreground mask** from **row crop RGB** images using depth estimation and histogram-based thresholding.

---

## Pipeline Overview

1. **Foreground Mask Extraction**

   * We use [Depth-Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) to generate depth maps from RGB images.
   * From the generated depth maps, we apply the following steps:

     1. **Histogram Computation** – compute the depth histogram for each map.
     2. **Valley Detection** – identify the minima (valleys) between significant modes in the histogram.
     3. **Thresholding** – select the optimal threshold based on the valley position and binarize the depth map.
     4. **Mask Generation** – the resulting binary mask separates the **foreground** from the **background**.


## To run the code:


2. **Clone our repository [Foreground-Mask-Extraction-In-Row-Crop-Images](https://github.com/djaliloh/Foreground-Mask-Extraction-In-Row-Crop-Images.git)**
    
    * Step 1:
        1. clone :  git clone https://github.com/djaliloh/Foreground-Mask-Extraction-In-Row-Crop-Images.git
        2. cd Foreground-Mask-Extraction-In-Row-Crop-Images

3. **Depth Estimation**

   * Step 2:
     1. Clone the official repository : git clone https://github.com/DepthAnything/Depth-Anything-V2.git
     2. Download the pre-trained weights (we use the `vitl` model).

4. **Run the script**
    * Step 3:
        1. Specify your data path in run_fgvegseg_pipline.sh file
        2. <pre> ``` bash run_fgvegseg_pipline.sh ``` </pre>



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


<!-- --- -->
