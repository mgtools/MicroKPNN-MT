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
conda env create -f requirements.yml -n microkpnn-mt
conda activate microkpnn-mt

# note: We use CUDA 11.7 and PyTorch 1.13. Please install the suitable PyTorch version to 
# your CUDA version (https://pytorch.org/get-started/locally/). 
```

**Step 3**: Run the following command for training, evaluation, and explanation: 

You can select your taxonomy based on the following: 

```python
{0:'superkingdom', 1:'phylum',2:'class', 3:'order', 4:'family', 5:'genus'}
```
Based on our experiments, we observed that, in general, the genus taxonomic rank yielded the best results. Therefore, we recommend using genus (--taxonomy 5); however, users are welcome to explore other taxonomic ranks for their analysis.

If you'd like to use a sample dataset to make the running procedure faster, please go to [Train on Sample User Dataset](#train-on-sample-user-dataset). To run the pipeline on the whole dataset use the following command: 

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

The format of the explanation results (`output/microkpnn_mt_fold0_explanation.pkl`) is shown in [Explanation Output Format](#explanation-output-format). The plots for explanation can be found in `exp_interpretation_plots.ipynb`. 

### Train on Sample User Dataset

Here we provide a smaller dataset containing 4 different phenotypes.

```bash
tar -xvzf ./Dataset/sample_relative_abundance.tar.gz -C ./Dataset/
```

```bash
# 5-fold validation
python MicroKPNN_MT.py \
--data_path Dataset/sample_relative_abundance.csv \
--metadata_path Dataset/sample_metadata.csv \
--output sample_output/ \
--taxonomy 5 \
--k_fold 5 \
--device 0

# note: To train only one model using randomly splitting training and test set, 
# please set k_fold as 0. 
```

### Explanation Output Format

```python
# 1. meta-data -> disease
'age_disease_explanations': {
    'disease_1': Tensor,  # Attributes for age to disease_1, size: (sample number, age class number)
    'disease_2': Tensor,  # Attributes for age to disease_2, size: (sample number, age class number)
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
    'age_1': Tensor,  # Attributes for hidden layers to age_1, size: (sample number, hidden nodes number)
    'age_2': Tensor,  # Attributes for hidden layers to age_2, size: (sample number, hidden nodes number)
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

## Our Experiments

This section offers the detailed commands we used in our experiments. 

**Experiment 1**: K-fold validation of disease prediction and missing meta-data prediction

```bash
# train and evluation (5-fold)
bash exp_5fold.sh
```

**Experiment 2**: k-fold validation to check the generalizability of our method:

We trained and tested our model on the following diseases: Cystic Fibrosis, Chronic Obstructive Pulmonary Disease, Bacterial Vaginosis, and healthy samples.

```bash
# train on train_dataset from 10 different projects and test and predict on test dataset from 6 independent projects.
bash exp_generalizability.sh
```

**Experiment 3**: Interpretation of the predictive models

```bash
# train, evaluate, and explain 20 models
bash exp_interpretation.sh
```
The codes for plotting explanation results are in `exp_interpretation_plots.ipynb`. 


## Benchmarking Models

We utilize well-known models such as **SVM**, **Random Forest**, and **XGBoost** for benchmarking. The repository includes two scripts tailored for different use cases:

- **`baseline_all_kfold.py`**: Predicts all metadata and disease labels using the specified models, providing a comprehensive evaluation across all targets.
```bash
  python benchmark/baseline_all_kfold.py \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--output output/ \
--model_type SVM
```
- **`baseline_disease_kfold.py`**: Focuses solely on predicting disease labels. This script is optimized for cases where only disease prediction results are needed, offering faster execution.

```bash
  python benchmark/baseline_disease_kfold.py \
--data_path Dataset/relative_abundance.csv \
--metadata_path Dataset/metadata.csv \
--output output/ \
--model_type SVM
 ```
