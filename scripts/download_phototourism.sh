# {brandenburg_gate, british_museum, lincoln_memorial_statue, pantheon_exterior, sacre_coeur, st_pauls_cathedral, taj_mahal,
SCENE=$1
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/${SCENE}.tar.gz
tar -xvf ${SCENE}.tar.gz
DATA_PATH=./data/phototourism
if [ ! -f "$DATA_PATH" ]; then
    mkdir data
    mkdir data/phototourism
fi
mv ${SCENE} ${DATA_PATH}
cp tsv/${SCENE}.tsv ${DATA_PATH}/${SCENE}
rm ${SCENE}.tar.gz
