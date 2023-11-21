SCENE=$1

python preprocess/save_dino_feature.py \
    --image_dir data/phototourism/${SCENE}/dense/images \
    --save_dir data/phototourism/${SCENE}/DINO \
    --tsv_path data/phototourism/${SCENE}/${SCENE}.tsv

FILE=./DPT/weights/dpt_large-midas-2f21e586.pt
if [ ! -f "$FILE" ]; then
    wget -c https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt -P ./DPT/weights
fi

python preprocess/save_dpt_depth.py \
    -i data/phototourism/${SCENE}/dense/images \
    -o data/phototourism/${SCENE}/DPT \
    --tsv_path data/phototourism/${SCENE}/${SCENE}.tsv \
    -t dpt_large
