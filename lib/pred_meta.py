import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score, f1_score

from dataset import MicroDataset
from model import MicroKPNN_MTL



def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def pred(model, loader, device): 
	model.eval()

	diseases = []
	pred_ages, pred_genders, pred_bmis, pred_bodysites, pred_diseases = [], [], [], [], []
	sample_ids = []

	with tqdm(total=len(loader)) as bar:
		for _, batch in enumerate(loader): 
			sample_id, spec, age, gender, bmi, bodysite, disease = batch
			spec = spec.type(torch.cuda.FloatTensor).to(device)
			age = age.type(torch.cuda.FloatTensor).to(device)
			gender = gender.type(torch.cuda.FloatTensor).to(device)
			bmi = bmi.type(torch.cuda.FloatTensor).to(device)
			bodysite = bodysite.type(torch.cuda.FloatTensor).to(device)
			disease = disease.type(torch.cuda.FloatTensor).to(device)
			invalid_disease = torch.isnan(disease)

			with torch.no_grad():
				pred_age, pred_gender, pred_bmi, pred_bodysite, pred_disease = model(spec, age, gender, bmi, bodysite)

			bar.set_description('Eval')
			bar.update(1)

			age_dim = age.size(1)
			pred_ages.append(pred_age.view(-1, age_dim).detach().cpu())

			pred_genders.append(pred_gender.detach().cpu())

			bmi_dim = bmi.size(1)
			pred_bmis.append(pred_bmi.view(-1, bmi_dim).detach().cpu())

			bodysite_dim = bodysite.size(1)
			pred_bodysites.append(pred_bodysite.view(-1, bodysite_dim).detach().cpu())

			disease_dim = disease.size(1)
			diseases.append(disease[~invalid_disease].view(-1, disease_dim).detach().cpu())
			pred_diseases.append(pred_disease.view(-1, disease_dim).detach().cpu())
			sample_ids = sample_ids + list(sample_id)

	pred_ages = torch.cat(pred_ages, dim=0)
	pred_genders = torch.cat(pred_genders, dim=0)
	pred_bmis = torch.cat(pred_bmis, dim=0)
	pred_bodysites = torch.cat(pred_bodysites, dim=0)
	diseases = torch.cat(diseases, dim = 0)
	pred_diseases = torch.cat(pred_diseases, dim=0)
	return sample_ids, pred_ages, pred_genders, pred_bmis, pred_bodysites, diseases, pred_diseases

def post_process_age(score, age_dict):
	re_dict = {v: k for k, v in age_dict.items()}
	score = np.argmax(score)
	return re_dict[score]

def post_process_gender(score, gender_dict):
	re_dict = {v: k for k, v in gender_dict.items()}
	score = np.argmax(score)
	return re_dict[score]

def post_process_bmi(score, bmi_dict):
	re_dict = {v: k for k, v in bmi_dict.items()}
	score = np.argmax(score)
	return re_dict[score]

def post_process_bodysite(score, bodysite_dict):
	print(score)
	print(type(score))
	print(bodysite_dict)
	re_dict = {v: k for k, v in bodysite_dict.items()}
	score = np.argmax(score)
	print(score)
	print(type(score))
	return re_dict[str(score)]

def post_process_disease(score, disease_dict):
	re_dict = {v: k for k, v in disease_dict.items()}
	score = np.argmax(score)
	return re_dict[str(score)]


if __name__ == "__main__":
	# Evalating settings
	parser = argparse.ArgumentParser(description='MicroKPNN Multi Task Learning (pred)')
	parser.add_argument('--k_fold', type=int, default=5, help='k for k-fold validation')

	parser.add_argument('--data_path', type=str, required=True, help='Path to data')
	parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata')
	parser.add_argument('--edge_list', type=str, required = True, help='Path to edge list')
	parser.add_argument('--output', type=str, required = True, help='Path to output')

	parser.add_argument('--checkpoint_path', type=str, default = '', help='Path to save checkpoint')
	parser.add_argument('--resume_path', type=str, default='', help='Path to pretrained model')
	parser.add_argument('--records_path', type=str, default='', help='Path to save records')
	parser.add_argument('--device', type=int, default=0, help='Which gpu to use if any (default: 0)')
	parser.add_argument('--seed', type=int, default=42, help='Seeds for random, torch, and numpy')
	args = parser.parse_args()

	# 0. Settings
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	# model settings: 
	metadatas = ['BMI', 'gender', 'age', 'bodysite', 'phenotype']
	edge_df = pd.read_csv(args.edge_list)
	parent_nodes = list(set(edge_df['parent'].tolist()))
	parent_nodes = [node for node in parent_nodes if node not in metadatas]
	df_relative_abundance = pd.read_csv(args.data_path, index_col = 0)
	species = df_relative_abundance.columns.values.tolist()
	print('species num:', len(species))

	metadata_df = pd.read_csv(args.metadata_path)
	print('meta-data:', list(metadata_df.columns))
	in_dim = len(species) # in_dim is fixed according to the edge list (the value will be checked when estalishing model)
	hidden_dim = len(parent_nodes)# hidden_dim is fixed according to the edge list (the value will be checked when estalishing model)
	mask = torch.zeros((in_dim, hidden_dim)) # this won't be saved in the checkpoint
	bodysite_num = len(list(set(metadata_df['BodySite'].values.tolist())))
	print('bodysite num:', bodysite_num)
	disease_num = len(list(set(metadata_df['disease'].values.tolist())))
	print('disease num:', disease_num)
	print(list(set(metadata_df['disease'].values.tolist())))
	
    # training settings: 
	lr = 0.0001
	batch_size = 64
	epoch_num = 100
	early_stop_step = 20
    
    # 1. Model 
	print('Establishing the model...')
	device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
	print(f'Device: {device}')
	model = MicroKPNN_MTL(species, args.edge_list, in_dim, hidden_dim,bodysite_num, disease_num, mask.to(device))
	species_dict = model.get_species_dict()
	num_params = sum(p.numel() for p in model.parameters())
	print(f'{str(model)} #Params: {num_params}')
	model.to(device)

	# 2. Data
	print('Loading the dataset...')
	dataset = MicroDataset(data_path=args.data_path, metadata_path=args.metadata_path, species_dict=species_dict, output=args.output + '/NetworkInput')
	age_dict = dataset.get_age_dict()
	gender_dict = dataset.get_gender_dict()
	bmi_dict = dataset.get_bmi_dict()
	bodysite_dict = dataset.get_bodysite_dict()
	disease_dict = dataset.get_disease_dict()

	all_loader = torch.utils.data.DataLoader(dataset,
											batch_size=batch_size,
											shuffle=True,
											num_workers=0,
											drop_last=True)

	# 3. Pred
	print('Loading the best model...')
	model.load_state_dict(torch.load(args.resume_path, map_location=device)['model_state_dict'])
	sample_ids, age_pred, gender_pred, bmi_pred, bodysite_pred, disease_true, disease_pred = pred(model, all_loader, device)
	
	_, disease_pred_tag = torch.max(disease_pred, dim=1)
	_, disease_true_tag = torch.max(disease_true, dim=1)
	
	disease_acc = accuracy_score(disease_true_tag, disease_pred_tag)
	# disease_auc = roc_auc_score(disease_true, disease_pred, average=None)
	# it need's all classes being available in y_true but here we have 5 classes available out of 26
	try:
		disease_auc = roc_auc_score(disease_true, disease_pred, average=None)
		disease_f1 = f1_score(disease_true_tag, disease_pred_tag, average=None)

	except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
		print('When calculating the AUC score for disease prediction, only one class is present in y_true: ')
		print(disease_true_tag)
		print('ROC AUC score is not defined in that case.')
		disease_auc = np.nan
		disease_f1 = np.nan
	print("Disease >>>\nACC: {}, F1: {}".format(disease_acc, disease_f1))
	records = {'Disease ACC': [], 'Disease AUC': [], 'Disease F1': []}
	records['Disease ACC'].append(disease_acc)
	records['Disease AUC'].append(disease_auc)
	records['Disease F1'].append(disease_f1)

	# final results
	records = pd.DataFrame.from_dict(records)
	print(records)
	records.to_csv(args.records_path + '/pred_eval.csv')
	print('save the records into {}'.format(args.records_path + '/pred_eval.csv'))
	print('Done!')
	
	print(len(sample_ids))
	age_pred = [post_process_age(i, age_dict) for i in age_pred.tolist()]
	gender_pred = [post_process_gender(i, gender_dict) for i in gender_pred.tolist()]
	bmi_pred = [post_process_bmi(i, bmi_dict) for i in bmi_pred.tolist()]
	bodysite_pred = [post_process_bodysite(i, bodysite_dict) for i in bodysite_pred.tolist()]
	disease_pred = [post_process_disease(i, disease_dict) for i in disease_pred.tolist()]

	# export the results
	data_df = pd.read_csv(args.metadata_path)
	res_df = pd.DataFrame({'ID': sample_ids, 'Pred age': age_pred, 'Pred sex': gender_pred, 'Pred BMI': bmi_pred, 'Pred bodysite': bodysite_pred, 'Pred disease':disease_pred})
	res_df = res_df.merge(data_df, how='left', left_on='ID', right_on='run_id')
	print(res_df)
	res_df.to_csv(args.records_path+'/pred_results.csv')
	print('Save the predicted metadata!')