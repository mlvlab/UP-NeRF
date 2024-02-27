ROOT=$1

python preprocess/save_dino_feature.py \
    --image_dir $ROOT/dense/images \
    --save_dir $ROOT/DINO \

FILE=./DPT/weights/dpt_large-midas-2f21e586.pt
if [ ! -f "$FILE" ]; then
    wget -c https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt -P ./DPT/weights
fi

python preprocess/save_dpt_depth.py \
    -i $ROOT/dense/images \
    -o $ROOT/DPT \
    -t dpt_large
