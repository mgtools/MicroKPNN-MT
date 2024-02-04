#! /bin/bash
# Train the model using 5-fold validation, and 
# inference on the whole dataset (no explanation). 

python lib/train_kfold.py --k_fold 5 \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--edge_list output/NetworkInput/EdgeList.csv \
--output output/ \
--checkpoint_path output/Checkpoint/microkpnn_mt.pt \
--records_path output/Record/microkpnn_mt.csv \
--device 0

for i in {0..4}
do
    python lib/pred_meta.py \
    --data_path Dataset/relative_abundance.csv \
    --metadata_path Dataset/metadata.csv \
    --edge_list output/NetworkInput/EdgeList.csv \
    --output output/ \
    --resume_path output/Checkpoint/microkpnn_mt_fold$i.pt \
    --records_path output/Record/ \
    --device 0 
done
