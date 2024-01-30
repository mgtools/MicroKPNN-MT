# MicroKPNN-MT

Multi-task microbiome-based knowledge-primed neural network

1. Unzip the datasets

```bash
tar -xvzf ./Dataset/relative_abundance.tar.gz -C ./Dataset/
tar -xvzf ./Default_Database/nodes.tar.gz -C ./Default_Database/
tar -xvzf ./Default_Database/names.tar.gz -C ./Default_Database/
```

2. Establish the anaconda environment

```bash
conda create -n microkpnn python=3.8
conda activate microkpnn

# install the suitable PyTorch version according to your CUDA version
# e.g.
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# install captum package for explanation
conda install captum -c conda-forge
pip install scikit-learn
```

2. Run the following command:  

You can select your taxonomy based on the following:

```python
{0:'superkingdom', 1:'phylum',2:'class', 3:'order', 4:'family', 5:'genus'}
```

To run the pipeline on the dataset use following command

```bash
# 5-fold validation
python MicroKPNN_MT.py \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--output output/ \
--taxonomy 5 \
--k_fold 5 \
--device 0

# note: To train only one model using randomly splitting training and test set, 
# please set k_fold as 0. 
```

Now if you wanna run one of your trained model on your dataset and get results you can do following:

```bash
python lib/pred_meta.py \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--edge_list output/NetworkInput/EdgeList.csv \
--output output/ \
--resume_path output/Checkpoint/microkpnn_mt_fold0.pt \
--records_path output/Record/ \
--explanation_path output/microkpnn_mt_fold0_explanation.pkl \
--device 0 

# note: `--explanation_path` is an optional parameter for explanation. 
```
