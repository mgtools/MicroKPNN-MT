# MicroKPNN-MT

Multi-task microbiome-based knowledge-primed neural network

## Dataset

We employed the mBodyMap database, featuring 34,233 preprocessed samples from 56 projects and 25 diseases. The dataset underwent categorization of BMI and age data, yielding insightful patterns for analysis. 

## Usage

**Step 1**: Clone the repo and unzip the datasets using the following commands: 

```bash
git clone https://github.com/mgtools/MicroKPNN-MT.git
cd MicroKPNN-MT

tar -xvzf ./Dataset/relative_abundance.tar.gz -C ./Dataset/
tar -xvzf ./Default_Database/nodes.tar.gz -C ./Default_Database/
tar -xvzf ./Default_Database/names.tar.gz -C ./Default_Database/
```

**Step 2**: Install anaconda from [here](https://docs.anaconda.com/free/anaconda/install/index.html) and establish the anaconda environment using the following commands: 

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

**Step 3**: Run the following command for training, evaluation, and explanation: 

You can select your taxonomy based on the following: 

```python
{0:'superkingdom', 1:'phylum',2:'class', 3:'order', 4:'family', 5:'genus'}
```
Based on our experiments, we observed that, in general, the genus taxonomic rank yielded the best results. Therefore, we recommend using genus (--taxonomy 5); however, users are welcome to explore other taxonomic ranks for their analysis.

To run the pipeline on the dataset use following command: 

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

Now if you want to run one of your trained models on your dataset and get results you can do following:

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

The format of the explanation results (`output/microkpnn_mt_fold0_explanation.pkl`) is:  

```python
# 1. meta-data -> disease
'age_disease_explanations': {
    'disease_1': Tensor,  # Attributes for age to disease_1, size: (sample_number, age_cls_number)
    'disease_2': Tensor,  # Attributes for age to disease_2, size: (sample_number, age_cls_number)
    ...
},
'gender_disease_explanations': {
    'disease_1': Tensor, 
    ...
},
'bmi_disease_explanations': {
    'disease_1': Tensor, 
    ...
},
'bodysite_disease_explanations': {
    'disease_1': Tensor, 
    ...
}, 
# 2. hidden layer -> meta-data
'hidden_age_explanations': {
    'age_1': Tensor,  # Attributes for hidden layers to age_1, size: (sample_number, hidden_nodes_number)
    'age_2': Tensor,  # Attributes for hidden layers to age_2, size: (sample_number, hidden_nodes_number)
    ...
}, 
'hidden_gender_explanations': {
    'gender_1': Tensor,
    ...
},
'hidden_bmi_explanations': {
    'bmi_1': Tensor,
    ...
},
'hidden_bodysite_explanations': {
    'bodysite_1': Tensor,
    ...
}
```

## Experiments

**Experiment 1**: K-fold validation of disease prediction and missing meta-data prediction

```bash
# train and evluation (5-fold)
bash exp_5fold.sh
```

**Experiment 2**: Interpretation of the predictive models

```bash
# train, evaluate, and explain 20 models
bash exp_interpretation.sh
```

The codes for plotting explanation results are in `exp_interpretation_plots.ipynb`. 

