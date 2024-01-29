# MicroKPNN-MT
Multi-task microbiome-based knowledge-primed neural network

First Unzip relative_abundance in Dataset directory

Then run following command:  

you can select your taxonomy based on the following:

```
{0:'superkingdom', 1:'phylum',2:'class', 3:'order', 4:'family', 5:'genus'}
```

to run the pipeline on the dataset use following command

```bash
python MicroKPNN_MT.py \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--output output/ \
--taxonomy 5 \
--device 0

```

now if you wanna run one of your trained model on your dataset and get results you can do following:

```bash
python lib/pred_meta.py \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--edge_list output/NetworkInput/EdgeList.csv \
--output output/ \
--resume_path output/Checkpoint/microkpnn_mt_fold0.pt \
--records_path output/Record/ \
--device 0 
```





