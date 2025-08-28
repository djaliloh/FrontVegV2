#!/bin/bash

# This script shell will run all our pipeline files
echo ">> Generate depth maps"
# python ForegroundVegSeg/utilis/generate_depth.py --img_path dataset --input_size 518 --outdir Depth-Anything-V2 --encoder vitl --pred_only --grayscale

echo ">> Thresholding"
python ForegroundVegSeg/utilis/tresholding.py




echo ">> End of the pipeline."