#! /bin/bash
# Train the model using 5-fold validation, and 
# inference on the whole dataset (no explanation). 

# python lib/train_meta_kfold.py --k_fold 5 \
# --data_path Dataset/relative_abundance.csv \
# --metadata_path Dataset/metadata.csv \
# --edge_list output/NetworkInput/EdgeList.csv \
# --output output/ \
# --checkpoint_path output/Checkpoint/microkpnn_mt.pt \
# --records_path output/Record/microkpnn_mt.csv \
# --device 0
python lib/train_meta_kfold.py --k_fold 5 \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--edge_list output/NetworkInput/EdgeList.csv \
--output output0524/ \
--checkpoint_path output0524/Checkpoint0524/microkpnn_mt_bs32.pt \
--records_path output0524/Record0524/microkpnn_mt_bs32.csv \
--device 0
python lib/train_meta_kfold.py --k_fold 5 \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--edge_list output/NetworkInput/EdgeList.csv \
--output output0524/ \
--checkpoint_path output0524/Checkpoint0715/microkpnn_mt.pt \
--records_path output0524/Record0715/microkpnn_mt.csv \
--device 0

# for i in {0..4}
# do
#     python lib/pred_meta.py \
#     --data_path Dataset/relative_abundance.csv \
#     --metadata_path Dataset/metadata.csv \
#     --edge_list output/NetworkInput/EdgeList.csv \
#     --output output/ \
#     --resume_path output/Checkpoint/microkpnn_mt_fold$i.pt \
#     --records_path output/Record/ \
#     --device 0 
# done
for i in {0..4}
do
    python lib/pred_meta.py \
    --data_path Dataset/relative_abundance.csv \
    --metadata_path Dataset/metadata.csv \
    --edge_list output/NetworkInput/EdgeList.csv \
    --output output0524/ \
    --resume_path output0524/Checkpoint0524/microkpnn_mt_fold$i.pt \
    --records_path output0524/Record0524/ \
    --device 0 
done

nohup python lib/baseline_all_kfold.py --k_fold 5 \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--model_type SVM \
--output output/ > svm.out

python lib/baseline_all_kfold.py --k_fold 5 \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--model_type RF \
--output output/

python lib/baseline_all_kfold.py --k_fold 5 \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--model_type XGBoost \
--output output/

# using the same embedding.pkl
nohup python -u lib/baseline_disease_kfold.py --k_fold 5 \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--model_type SVM \
--output output/ > svm_disease.out

nohup python -u lib/baseline_disease_kfold.py --k_fold 5 \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--model_type RF \
--output output/ > rf_disease.out

nohup python -u lib/baseline_disease_kfold.py --k_fold 5 \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--model_type XGBoost \
--output output/ > xgboost_disease.out

