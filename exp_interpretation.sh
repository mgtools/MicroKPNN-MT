#! /bin/bash

# Train the model 20 times in parallel
# note: the outputs are crazy, please leave them away
for i in {0..9}
do
    echo "python lib/train_meta.py --data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--edge_list output/NetworkInput/genus/EdgeList.csv \
--output output/ \
--checkpoint_path output/Checkpoint/microkpnn_$i.pt \
--device 0"
    python lib/train_meta.py --data_path Dataset/relative_abundance.csv \
    --metadata_path Dataset/metadata.csv \
    --edge_list output/NetworkInput/genus/EdgeList.csv \
    --output output/ \
    --checkpoint_path output/Checkpoint/microkpnn_$i.pt \
    --device 0 &
done

for i in {10..19}
do
    echo "python lib/train_meta.py --data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--edge_list output/NetworkInput/genus/EdgeList.csv \
--output output/ \
--checkpoint_path output/Checkpoint/microkpnn_$i.pt \
--resume_path output/Checkpoint/microkpnn_$i.pt \
--device 1"
    python lib/train_meta.py --data_path Dataset/relative_abundance.csv \
    --metadata_path Dataset/metadata.csv \
    --edge_list output/NetworkInput/genus/EdgeList.csv \
    --output output/ \
    --checkpoint_path output/Checkpoint/microkpnn_$i.pt \
    --resume_path output/Checkpoint/microkpnn_$i.pt \
    --device 1 &
done

wait

# Explain them one by one
for i in {0..19}
do
    echo "python lib/explain_meta.py --data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--edge_list output/NetworkInput/genus/EdgeList.csv \
--output output/ \
--resume_path output/Checkpoint/microkpnn_$i.pt \
--explanation_path output/Explain/explanation_sample_$i.pkl \
--device 0"
    python lib/explain_meta.py --data_path Dataset/relative_abundance.csv \
    --metadata_path Dataset/metadata.csv \
    --edge_list output/NetworkInput/genus/EdgeList.csv \
    --output output/ \
    --resume_path output/Checkpoint/microkpnn_$i.pt \
    --explanation_path output/Explain/explanation_sample_$i.pkl \
    --device 0
done
