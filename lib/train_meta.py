import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score, f1_score

from dataset import MicroDataset
from model import MicroKPNN_MTL



def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def train_step(model, loader, optimizer, device): 
	criterion_age = nn.CrossEntropyLoss()
	criterion_gender = nn.BCELoss()
	criterion_bmi = nn.CrossEntropyLoss()
	criterion_bodysite = nn.CrossEntropyLoss()
	criterion_disease = nn.CrossEntropyLoss()

	with tqdm(total=len(loader)) as bar:
		for step, batch in enumerate(loader): 
			_, spec, age, gender, bmi, bodysite, disease = batch

			batch_size = len(age)
			age_size = max(1,torch.sum(~torch.isnan(age)).item())
			gender_size = max(1, torch.sum(~torch.isnan(gender)).item())
			bmi_size = max(1, torch.sum(~torch.isnan(bmi)).item())
			bodysite_size = max(1, torch.sum(~torch.isnan(bodysite)).item())

			spec = spec.type(torch.cuda.FloatTensor).to(device)

			age = age.type(torch.cuda.FloatTensor).to(device)
			invalid_age = torch.isnan(age)
			age_dim = age.size(1)

			gender = gender.type(torch.cuda.FloatTensor).to(device)
			invalid_gender = torch.isnan(gender)

			bmi = bmi.type(torch.cuda.FloatTensor).to(device)
			invalid_bmi = torch.isnan(bmi)
			bmi_dim = bmi.size(1)
			
			bodysite = bodysite.type(torch.cuda.FloatTensor).to(device)
			invalid_bodysite = torch.isnan(bodysite)
			bodysite_dim = bodysite.size(1)

			disease = disease.type(torch.cuda.FloatTensor).to(device)
			invalid_disease = torch.isnan(disease)
			disease_dim = disease.size(1)
			
			optimizer.zero_grad()
			model.train()
			
			pred_age, pred_gender, pred_bmi, pred_bodysite, pred_disease = model(spec, age, gender, bmi, bodysite) 
			
			loss = criterion_age(pred_age[~invalid_age].view(-1, age_dim), age[~invalid_age].view(-1, age_dim)) * (batch_size/age_size) + \
					criterion_gender(pred_gender[~invalid_gender], gender[~invalid_gender]) * (batch_size/gender_size) + \
					criterion_bmi(pred_bmi[~invalid_bmi].view(-1, bmi_dim), bmi[~invalid_bmi].view(-1, bmi_dim)) * (batch_size/bmi_size) + \
					criterion_bodysite(pred_bodysite[~invalid_bodysite].view(-1, bodysite_dim), bodysite[~invalid_bodysite].view(-1, bodysite_dim)) * (batch_size/bodysite_size) + \
					criterion_disease(pred_disease[~invalid_disease].view(-1, disease_dim), disease[~invalid_disease].view(-1, disease_dim)) 
					
			
			loss.backward()

			bar.set_description('Train')
			bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
			bar.update(1)

			optimizer.step()
	return 

def eval_step(model, loader, device): 
	model.eval()

	ages = []
	pred_ages = []

	genders = []
	pred_genders = []

	bmis = []
	pred_bmis = []

	bodysites = []
	pred_bodysites = []

	diseases = []
	pred_diseases = []

	sample_ids = []
	with tqdm(total=len(loader)) as bar:
		for _, batch in enumerate(loader): 

			sample_id, spec, age, gender, bmi, bodysite, disease = batch

			spec = spec.type(torch.cuda.FloatTensor).to(device)

			age = age.type(torch.cuda.FloatTensor).to(device)
			invalid_age = torch.isnan(age)

			gender = gender.type(torch.cuda.FloatTensor).to(device)
			invalid_gender = torch.isnan(gender)

			bmi = bmi.type(torch.cuda.FloatTensor).to(device)
			invalid_bmi = torch.isnan(bmi)

			bodysite = bodysite.type(torch.cuda.FloatTensor).to(device)
			invalid_bodysite = torch.isnan(bodysite)

			disease = disease.type(torch.cuda.FloatTensor).to(device)
			invalid_disease = torch.isnan(disease)

			with torch.no_grad():
				pred_age, pred_gender, pred_bmi, pred_bodysite, pred_disease = model(spec, age, gender, bmi, bodysite)

			bar.set_description('Eval')
			bar.update(1)

			age_dim = age.size(1)
			ages.append(age[~invalid_age].view(-1, age_dim).detach().cpu())
			pred_ages.append(pred_age[~invalid_age].view(-1, age_dim).detach().cpu())

			genders.append(gender[~invalid_gender].detach().cpu())
			pred_genders.append(pred_gender[~invalid_gender].detach().cpu())

			bmi_dim = bmi.size(1)
			bmis.append(bmi[~invalid_bmi].view(-1, bmi_dim).detach().cpu())
			pred_bmis.append(pred_bmi[~invalid_bmi].view(-1, bmi_dim).detach().cpu())

			bodysite_dim = bodysite.size(1)
			bodysites.append(bodysite[~invalid_bodysite].view(-1, bodysite_dim).detach().cpu())
			pred_bodysites.append(pred_bodysite[~invalid_bodysite].view(-1, bodysite_dim).detach().cpu())

			disease_dim = disease.size(1)
			diseases.append(disease[~invalid_disease].view(-1, disease_dim).detach().cpu())
			pred_diseases.append(pred_disease[~invalid_disease].view(-1, disease_dim).detach().cpu())


			sample_ids = sample_ids + list(sample_id)

	ages = torch.cat(ages, dim = 0)
	pred_ages = torch.cat(pred_ages, dim = 0)

	genders = torch.cat(genders, dim = 0)
	pred_genders = torch.cat(pred_genders, dim = 0)

	bmis = torch.cat(bmis, dim = 0)
	pred_bmis = torch.cat(pred_bmis, dim = 0)

	bodysites = torch.cat(bodysites, dim = 0)
	pred_bodysites = torch.cat(pred_bodysites, dim = 0)

	diseases = torch.cat(diseases, dim = 0)
	pred_diseases = torch.cat(pred_diseases, dim = 0)

	return sample_ids, ages, pred_ages, genders, pred_genders, bmis, pred_bmis, bodysites, pred_bodysites, diseases, pred_diseases



if __name__ == "__main__":
	# Training settings
	parser = argparse.ArgumentParser(description='MicroKPNN Multi Task Learning')
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

	# Create the output directory 
	try: 
		os.makedirs(args.output, exist_ok = True) 
		print("Directory '%s' created successfully" % args.output) 
	except OSError as error: 
		print("Directory '%s' can not be created" % args.output)

	# 0. Settings
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

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
	in_dim = len(species) 
	hidden_dim = len(parent_nodes)
	mask = torch.zeros((in_dim, hidden_dim)) # this won't be saved in the checkpoint
	bodysite_num = len(list(set(metadata_df['BodySite'].values.tolist())))
	print('bodysite num:', bodysite_num)
	disease_num = len(list(set(metadata_df['disease'].values.tolist())))
	print('disease num:', disease_num)

	# training settings: 
	lr = 0.001
	batch_size = 16
	epoch_num = 100
	early_stop_step = 20
	early_stop_patience = 0
	best_disease_acc = 0
	
	# 1. Model 
	print('Establishing the model...')
	device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
	print(f'Device: {device}')
	model = MicroKPNN_MTL(species, args.edge_list, in_dim, hidden_dim, bodysite_num, disease_num, mask.to(device))
	species_dict = model.get_species_dict()
	num_params = sum(p.numel() for p in model.parameters())
	print(f'{str(model)} #Params: {num_params}')
	model.to(device)

	# 2. Data
	print('Loading the dataset...')
	dataset = MicroDataset(data_path=args.data_path, metadata_path=args.metadata_path, species_dict=species_dict, output= args.output)
	assert len(dataset.get_disease_dict()) - 1 == disease_num, "Setting and metadata are not match, disease_num={}, \
															but there are {} diseases in metadata".format(disease_num, len(dataset.get_disease_dict)-1)
	
	
	train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(args.seed)) # set the manual_seed making sure the dataset is split equally every time
	print('# Train: {}, # Val: {}'.format(len(train_set), len(val_set)))

	train_loader = torch.utils.data.DataLoader(train_set,
												batch_size=batch_size,
												shuffle=True,
												num_workers=0,
												drop_last=True)
	val_loader = torch.utils.data.DataLoader(val_set,
												batch_size=batch_size,
												shuffle=False,
												num_workers=0,
												drop_last=True)

	# 3. Train
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
	# load the checkpoints
	if args.resume_path != '':
		print("Load the checkpoints...")
		epoch_start = torch.load(args.resume_path, map_location=device)['epoch']
		model.load_state_dict(torch.load(args.resume_path, map_location=device)['model_state_dict'])
		optimizer.load_state_dict(torch.load(args.resume_path, map_location=device)['optimizer_state_dict'])
		scheduler.load_state_dict(torch.load(args.resume_path, map_location=device)['scheduler_state_dict'])
	else:
		epoch_start = 1

	for epoch in range(epoch_start, epoch_num+1): 
		print('\nEpoch {}'.format(epoch))
		train_step(model, train_loader, optimizer, device)

		sample_ids, age_true, age_pred, gender_true, gender_pred, bmi_true, bmi_pred, bodysite_true, bodysite_pred, disease_true, disease_pred = eval_step(model, val_loader, device)
		
		# age
		_, age_true_tag = torch.max(age_true, dim=1)
		_, age_pred_tag = torch.max(age_pred, dim=1)
		age_acc = accuracy_score(age_true_tag, age_pred_tag)
		try:
			age_auc = roc_auc_score(age_true, age_pred, average=None)
		except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case. 
			print('When calculating the AUC score for age prediction, only one class is present in y_true: ')
			print(age_true_tag)
			print('ROC AUC score is not defined in that case.')
			age_auc = np.nan

		print("Age >>>\nACC: {}, AUC: {}".format(age_acc, age_auc))
		
		# gender
		gender_pred_tag = [1 if p > 0.5 else 0 for p in gender_pred]
		gender_acc = accuracy_score(gender_true, gender_pred_tag)
		gender_auc = roc_auc_score(gender_true, gender_pred)
		print("Gender >>>\nACC: {}, AUC: {}".format(gender_acc, gender_auc))
		
		# BMI
		_, bmi_true_tag = torch.max(bmi_true, dim=1)
		_, bmi_pred_tag = torch.max(bmi_pred, dim=1)
		bmi_acc = accuracy_score(bmi_true_tag, bmi_pred_tag)
		print("BMI >>>\nAcc: {}".format(bmi_acc))

		# bodysite
		_, bodysite_true_tag = torch.max(bodysite_true, dim=1)
		_, bodysite_pred_tag = torch.max(bodysite_pred, dim=1)
		bodysite_acc = accuracy_score(bodysite_true_tag, bodysite_pred_tag)
		try:
			bodysite_auc = roc_auc_score(bodysite_true, bodysite_pred, average=None)
		except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case. 
			print('When calculating the AUC score for bodysite prediction, only one class is present in y_true: ')
			print(bodysite_true_tag)
			print('ROC AUC score is not defined in that case.')
			bodysite_auc = np.nan

		print("Bodysite >>>\nACC: {}, AUC: {}".format(bodysite_acc, bodysite_auc))
	

		# disease
		_, disease_true_tag = torch.max(disease_true, dim=1)
		_, disease_pred_tag = torch.max(disease_pred, dim=1)
		disease_acc = accuracy_score(disease_true_tag, disease_pred_tag)
		try:
			disease_auc = roc_auc_score(disease_true, disease_pred, average=None)
			disease_f1 = f1_score(disease_true_tag, disease_pred_tag, average=None)
		except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case. 
			print('When calculating the AUC score for disease prediction, only one class is present in y_true: ')
			print(disease_true_tag)
			print('ROC AUC score is not defined in that case.')
			disease_auc = np.nan
			disease_f1 = np.nan
		print("Disease >>>\nACC: {}, AUC: {}, F1: {}".format(disease_acc, disease_auc, disease_f1))
		

		if disease_acc > best_disease_acc: 
			best_disease_acc = disease_acc

			if args.checkpoint_path != '':
				print('Saving checkpoint...')
				checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_disease_acc': best_disease_acc, 'num_params': num_params}
				torch.save(checkpoint, args.checkpoint_path)

			early_stop_patience = 0
			print('Early stop patience reset')
		else:
			early_stop_patience += 1
			print('Early stop count: {}/{}'.format(early_stop_patience, early_stop_step))

		scheduler.step(disease_acc) # ReduceLROnPlateau
		print(f'Best disease accuracy so far: {disease_acc}')

		if early_stop_patience == early_stop_step: 
			print('Early stop!')
			break

	# final results
	print('Loading the best model...')
    records = {'Age ACC': [], 'Age AUC': [], 'Age F1': [],
				'Gender ACC': [], 'Gender AUC': [], 'Gender F1': [],
				'BMI ACC': [], 'BMI AUC': [], 'BMI F1': [],
				'Bodysite ACC': [], 'Bodysite AUC': [], 'Bodysite F1': [],
				'Disease ACC': [], 'Disease AUC': [], 'Disease F1': []}
	model.load_state_dict(torch.load(args.checkpoint_path, map_location=device)['model_state_dict'])
	sample_ids, age_true, age_pred, gender_true, gender_pred, bmi_true, bmi_pred, bodysite_true, bodysite_pred, disease_true, disease_pred = eval_step(model, val_loader, device)
	
	# age
    _, age_pred_tag = torch.max(age_pred, dim=1)
    _, age_true_tag = torch.max(age_true, dim=1)
    age_acc = accuracy_score(age_true_tag, age_pred_tag)
    try:
        age_auc = roc_auc_score(age_true, age_pred, average=None)
        age_f1 = f1_score(age_true_tag, age_pred_tag, average=None)
    except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
        print('When calculating the AUC score for age prediction, only one class is present in y_true: ')
        print(age_true_tag)
        print('ROC AUC score is not defined in that case.')
        age_auc = np.nan
        age_f1 = np.nan
    print("Age >>>\nACC: {}, AUC: {}".format(age_acc, age_auc))

    
    # gender
    gender_pred_binary = [1 if p > 0.5 else 0 for p in gender_pred]
    gender_acc = accuracy_score(gender_true, gender_pred_binary)
    try:
        gender_auc = roc_auc_score(gender_true, gender_pred, average=None)
        gender_f1 = f1_score(gender_true, gender_pred_binary, average=None)
    except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
        print('When calculating the AUC score for age prediction, only one class is present in y_true: ')
        print(gender_true)
        print('ROC AUC score is not defined in that case.')
        gender_auc = np.nan
        gender_f1 = np.nan
    
    print("Gender >>>\nACC: {}, AUC: {}".format(gender_acc, gender_auc))
    
    # BMI
    _, bmi_true_tag = torch.max(bmi_true, dim=1)
    _, bmi_pred_tag = torch.max(bmi_pred, dim=1)
    bmi_acc = accuracy_score(bmi_true_tag, bmi_pred_tag)
    try:
        bmi_auc = roc_auc_score(bmi_true, bmi_pred, average=None)
        bmi_f1 = f1_score(bmi_true_tag, bmi_pred_tag, average=None)
    except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
        print('When calculating the AUC score for age prediction, only one class is present in y_true: ')
        print(bmi_true)
        print('ROC AUC score is not defined in that case.')
        bmi_auc = np.nan
        bmi_f1 = np.nan
    print("BMI >>>\nAcc: {}, AUC".format(bmi_acc, bmi_auc))

    # bodysite
    _, bodysite_pred_tag = torch.max(bodysite_pred, dim=1)
    _, bodysite_true_tag = torch.max(bodysite_true, dim=1)
    bodysite_acc = accuracy_score(bodysite_true_tag, bodysite_pred_tag)
    try:
        bodysite_auc = roc_auc_score(bodysite_true, bodysite_pred, average=None)
        bodysite_f1 = f1_score(bodysite_true_tag, bodysite_pred_tag, average=None)
    except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
        print('When calculating the AUC score for bodysite prediction, only one class is present in y_true: ')
        print(bodysite_true_tag)
        print('ROC AUC score is not defined in that case.')
        bodysite_auc = np.nan
        bodysite_f1 = np.nan
    print("Bodysite >>>\nACC: {}, AUC: {}".format(bodysite_acc, bodysite_auc))


    # disease
    _, disease_pred_tag = torch.max(disease_pred, dim=1)
    _, disease_true_tag = torch.max(disease_true, dim=1)
    disease_acc = accuracy_score(disease_true_tag, disease_pred_tag)
    try:
        disease_auc = roc_auc_score(disease_true, disease_pred, average=None)
        disease_f1 = f1_score(disease_true_tag, disease_pred_tag, average=None)
    except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
        print('When calculating the AUC score for disease prediction, only one class is present in y_true: ')
        print(disease_true_tag)
        print('ROC AUC score is not defined in that case.')
        disease_auc = np.nan
        disease_f1 = np.nan
    print("Disease >>>\nACC: {}, AUC: {}, F1: {}".format(disease_acc, disease_auc, disease_f1))


	# disease
	_, disease_pred_tag = torch.max(disease_pred, dim=1)
	_, disease_true_tag = torch.max(disease_true, dim=1)
	disease_acc = accuracy_score(disease_true_tag, disease_pred_tag)
	try:
		disease_auc = roc_auc_score(disease_true, disease_pred, average=None)
		disease_f1 = f1_score(disease_true_tag, disease_pred_tag, average=None)
	except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
		print('When calculating the AUC score for disease prediction, only one class is present in y_true: ')
		print(disease_true_tag)
		print('ROC AUC score is not defined in that case.')
		disease_auc = np.nan
		disease_f1 = np.nan
	print("Disease >>>\nACC: {}, AUC: {}, F1: {}".format(disease_acc, disease_auc, disease_f1))
	

    # update the records
    records['Age ACC'].append(age_acc)
    records['Age AUC'].append(age_auc)
    records['Age F1'].append(age_f1)
    records['Gender ACC'].append(gender_acc)
    records['Gender AUC'].append(gender_auc)
    records['Gender F1'].append(gender_f1)
    records['BMI ACC'].append(bmi_acc)
    records['BMI AUC'].append(bmi_auc)
    records['BMI F1'].append(bmi_f1)
    records['Bodysite ACC'].append(bodysite_acc)
    records['Bodysite AUC'].append(bodysite_auc)
    records['Bodysite F1'].append(bodysite_f1)
    records['Disease ACC'].append(disease_acc)
    records['Disease AUC'].append(disease_auc)
    records['Disease F1'].append(disease_f1)
		
	records = pd.DataFrame.from_dict(records)
	print(records)
	records.to_csv(args.records_path)
	print('save the records into {}'.format(args.records_path))
	print('Done!')