#!/bin/bash

### This script shell will run all our pipeline files ###
echo ">> Pipeline start..."

# timestamp - start
start=$(date +%s)
echo "- START : $(date -d @$start) -"


######## Paths and prompts ######## 
# datapath="/media/utilisateur/DATA/DJALIL/data_peachdisease"
#prompts=("hypertrophied leaves ." "swollen leaves ." "curled leaves ." "distorted foliage ." "blistered leaves ." "puffy leaves ." "malformed leaves .")
#prompts=("black rot leaf." "Red discolored leaf ." "swollen leaf ." "hypertrophied leaf ." "malformed leaves .")
datapath="/media/utilisateur/DATA/DJALIL/test-data/Tiff_corrige"  #"/media/utilisateur/DATA/DJALIL/data_winedisease"
prompts=("leaves close to camera ." "leaves in front of camera .")


# ########### 
echo -e "----------\n >> Generate depth maps"
python ForegroundVegSeg/utilis/generate_depth.py \
    --img_path "$datapath" \
    --input_size 518 \
    --outdir Depth-Anything-V2 \
    --encoder vitl \
    --pred_only --grayscale


echo -e "----------\n >> Thresholding"
python ForegroundVegSeg/utilis/tresholding.py


echo -e "----------\n >> Batch segmentation with Dino2 + Sam2"
python ForegroundVegSeg/utilis/batch_segmentation.py \
    --input_path "$datapath" \
    --output_path dinoSam2-outputs \
    --text_prompts "${prompts[@]}" #"$prompts" #

    
echo -e "----------\n >> Intersection"
python ForegroundVegSeg/utilis/AND.py \
    --fgm_path opt_min-outputs \
    --fgvm_path dinoSam2-outputs \
    --output_path AND_oper-outputs \
    --text_prompts "${prompts[@]}"


echo -e "----------\n >> Superposition" 
python ForegroundVegSeg/utilis/overlay.py \
    --RGB_path "$datapath" \
    --AND_path AND_oper-outputs \
    --output_path Superpose-outputs \
    --text_prompts "${prompts[@]}"
# ###########


# timestamp - end
end=$(date +%s)
echo "----------\n >> END : $(date -d @$end)"
duration=$((end - start)) # duration
H=$((duration / 3600))  # heures               
M=$(((duration % 3600) / 60)) #minutes
echo ">> Durée : ${H}h ${M}m ${duration}s"

echo "- End of the pipeline. -"












