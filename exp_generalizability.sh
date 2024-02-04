#! /bin/bash
# Train the model using 5-fold validation on train dataset 
# and run the trained models on the test dataset (no explanation).

python lib/train_meta_kfold.py --k_fold 5 \
--data_path Dataset/train_relative_abundance.csv \
--metadata_path Dataset/train_metadata.csv \
--edge_list train_output/NetworkInput/EdgeList.csv \
--output train_output/ \
--checkpoint_path train_output/Checkpoint/microkpnn_mt_train.pt \
--records_path train_output/Record/microkpnn_mt_train.csv \
--device 0

for i in {0..4}
do
    python lib/pred_meta.py \
    --data_path Dataset/test_relative_abundance.csv \
    --metadata_path Dataset/test_metadata.csv \
    --edge_list train_output/NetworkInput/EdgeList.csv \
    --output train_output/ \
    --resume_path train_output/Checkpoint/microkpnn_mt_train_fold$i.pt \
    --records_path train_output/Record \
    --device 0
    mv train_output/Record/pred_eval.csv train_output/Record/test_pred_eval_fold$i.csv
    mv train_output/Record/pred_results.csv train_output/Record/test_pred_results_fold$i.csv
done
