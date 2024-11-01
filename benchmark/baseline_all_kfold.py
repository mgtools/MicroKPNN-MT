import os
import argparse
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
import joblib
import time

from dataset import MicroDataset
from torch.utils.data import RandomSampler, DataLoader, Subset



def train_model(X_train, y_train, checkpoint_path, model_type, seed): 
	if model_type == 'SVM': 
		model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=seed)
	elif model_type == 'RF': 
		model = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=8)
	elif model_type == 'XGBoost':
		model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.01, random_state=seed, n_jobs=-1)
	model.fit(X_train, y_train)
	joblib.dump(model, checkpoint_path)
	return model

def eval_model(model, X_test, y_test):
	y_pred = model.predict_proba(X_test)
	return y_test, y_pred

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='Baseline Models Training')
	parser.add_argument('--k_fold', type=int, default=5, help='Number of folds for cross-validation')
	parser.add_argument('--data_path', type=str, required=True, help='Path to data')
	parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata')
	parser.add_argument('--model_type', type=str, required=True, help='SVM, RF or XGBoost?')
	parser.add_argument('--output', type=str, required=True, help='Path to output')
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	args = parser.parse_args()

	# Set random seed
	np.random.seed(args.seed)

	# Load the dataset
	print('Loading the dataset...')
	df_relative_abundance = pd.read_csv(args.data_path, index_col = 0)
	species = df_relative_abundance.columns.values.tolist()
	print('species num:', len(species))
	species_dict = {k: i for i, k in enumerate(species)}
	dataset = MicroDataset(data_path=args.data_path, 
							metadata_path=args.metadata_path, 
							species_dict=species_dict, 
							output=args.output +'/NetworkInput/')

	# Split the data into k-folds
	indices = list(range(len(dataset)))
	kfold = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=42)

	# Initialize records
	records = {'Age ACC': [], 'Age AUC': [], 'Age F1': [], 'Age AUPRC': [],
			   'Gender ACC': [], 'Gender AUC': [], 'Gender F1': [], 'Gender AUPRC': [],
			   'BMI ACC': [], 'BMI AUC': [], 'BMI F1': [], 'BMI AUPRC': [],
			   'Bodysite ACC': [], 'Bodysite AUC': [], 'Bodysite F1': [], 'Bodysite AUPRC': [],
			   'Disease ACC': [], 'Disease AUC': [], 'Disease F1': [], 'Disease AUPRC': [], 
			   'Fold Index': []}

	# Iterate over each fold
	for fold_idx, (train_indices, valid_indices) in enumerate(kfold.split(indices, dataset.diseases_idx)): 
		print(f'\nFold {fold_idx + 1}')

		# Initialize models and checkpoint paths for each task
		age_model = None
		gender_model = None
		bmi_model = None
		bodysite_model = None
		disease_model = None

		age_checkpoint_path = os.path.join(args.output, 'Checkpoint_Oth', f'{args.model_type}_age_model_fold{fold_idx + 1}.joblib')
		gender_checkpoint_path = os.path.join(args.output, 'Checkpoint_Oth', f'{args.model_type}_gender_model_fold{fold_idx + 1}.joblib')
		bmi_checkpoint_path = os.path.join(args.output, 'Checkpoint_Oth', f'{args.model_type}_bmi_model_fold{fold_idx + 1}.joblib')
		bodysite_checkpoint_path = os.path.join(args.output, 'Checkpoint_Oth', f'{args.model_type}_bodysite_model_fold{fold_idx + 1}.joblib')
		disease_checkpoint_path = os.path.join(args.output, 'Checkpoint_Oth', f'{args.model_type}_disease_model_fold{fold_idx + 1}.joblib')
		print('Check point paths:\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(age_checkpoint_path, 
																		gender_checkpoint_path, 
																		bmi_checkpoint_path, 
																		bodysite_checkpoint_path, 
																		disease_checkpoint_path))

		# Prepare train and validation data for each task
		X_train = []
		y_train_age, y_train_gender, y_train_bmi, y_train_bodysite, y_train_disease = [], [], [], [], []

		for idx in train_indices:
			sample_id, spec, age, gender, bmi, bodysite, disease = dataset[idx]
			X_train.append(spec)
			y_train_age.append(age)
			y_train_gender.append(gender)
			y_train_bmi.append(bmi)
			y_train_bodysite.append(bodysite)
			y_train_disease.append(disease)

		X_train = np.stack(X_train, axis=0)
		y_train_age = np.stack(y_train_age, axis=0).argmax(axis=1)
		y_train_gender = np.stack(y_train_gender, axis=0).ravel()
		y_train_bmi = np.stack(y_train_bmi, axis=0).argmax(axis=1)
		y_train_bodysite = np.stack(y_train_bodysite, axis=0).argmax(axis=1)
		y_train_disease = np.stack(y_train_disease, axis=0).argmax(axis=1)

		# Train models for each task
		print('Training model for age prediction...')
		age_model = train_model(X_train[~np.isnan(y_train_age)], y_train_age[~np.isnan(y_train_age)], 
								age_checkpoint_path, args.model_type, args.seed)
		print('Training model for gender prediction...')
		gender_model = train_model(X_train[~np.isnan(y_train_gender)], y_train_gender[~np.isnan(y_train_gender)], 
								gender_checkpoint_path, args.model_type, args.seed)
		print('Training model for BMI prediction...')
		bmi_model = train_model(X_train[~np.isnan(y_train_bmi)], y_train_bmi[~np.isnan(y_train_bmi)], 
								bmi_checkpoint_path, args.model_type, args.seed)
		print('Training model for bodysite prediction...')
		bodysite_model = train_model(X_train[~np.isnan(y_train_bodysite)], y_train_bodysite[~np.isnan(y_train_bodysite)], 
								bodysite_checkpoint_path, args.model_type, args.seed)
		print('Training model for disease prediction...')
		disease_model = train_model(X_train[~np.isnan(y_train_disease)], y_train_disease[~np.isnan(y_train_disease)], 
								disease_checkpoint_path, args.model_type, args.seed)

		# Prepare validation data
		X_valid = []
		y_valid_age, y_valid_gender, y_valid_bmi, y_valid_bodysite, y_valid_disease = [], [], [], [], []
		sample_ids_valid = []

		for idx in valid_indices:
			_, spec, age, gender, bmi, bodysite, disease = dataset[idx]
			X_valid.append(spec)
			y_valid_age.append(age)
			y_valid_gender.append(gender)
			y_valid_bmi.append(bmi)
			y_valid_bodysite.append(bodysite)
			y_valid_disease.append(disease)
			sample_ids_valid.append(sample_id)

		X_valid = np.stack(X_valid, axis=0)
		y_valid_age = np.stack(y_valid_age, axis=0)
		y_valid_gender = np.stack(y_valid_gender, axis=0)
		y_valid_bmi = np.stack(y_valid_bmi, axis=0)
		y_valid_bodysite = np.stack(y_valid_bodysite, axis=0)
		y_valid_disease = np.stack(y_valid_disease, axis=0)

		# Evaluate models for each task
		print('Evaluating models...')
		y_true_age, y_pred_age = eval_model(age_model, X_valid[~np.isnan(y_valid_age).any(axis=1)], y_valid_age[~np.isnan(y_valid_age).any(axis=1)])
		y_true_gender, y_pred_gender = eval_model(gender_model, X_valid[~np.isnan(y_valid_gender).any(axis=1)], y_valid_gender[~np.isnan(y_valid_gender).any(axis=1)])
		y_true_bmi, y_pred_bmi = eval_model(bmi_model, X_valid[~np.isnan(y_valid_bmi).any(axis=1)], y_valid_bmi[~np.isnan(y_valid_bmi).any(axis=1)])
		y_true_bodysite, y_pred_bodysite = eval_model(bodysite_model, X_valid[~np.isnan(y_valid_bodysite).any(axis=1)], y_valid_bodysite[~np.isnan(y_valid_bodysite).any(axis=1)])
		y_true_disease, y_pred_disease = eval_model(disease_model, X_valid[~np.isnan(y_valid_disease).any(axis=1)], y_valid_disease[~np.isnan(y_valid_disease).any(axis=1)])

		# Convert predicted probabilities to strings
		y_pred_age_str = [','.join(map(str, prob)) for prob in y_pred_age]
		y_pred_gender_str = [','.join(map(str, prob)) for prob in y_pred_gender]
		y_pred_bmi_str = [','.join(map(str, prob)) for prob in y_pred_bmi]
		y_pred_bodysite_str = [','.join(map(str, prob)) for prob in y_pred_bodysite]
		y_pred_disease_str = [','.join(map(str, prob)) for prob in y_pred_disease]

		# Save predictions and true values for each task
		pd.DataFrame({'sample_id': [sample_ids_valid[i] for i in range(len(sample_ids_valid)) if ~np.isnan(y_valid_age[i]).any()],
					  'true': y_true_age.argmax(axis=1).tolist(), 
					  'pred_prob': y_pred_age_str}).to_csv(os.path.join(args.output, 'Checkpoint_Oth', f'{args.model_type}_age_fold{fold_idx + 1}_predictions.csv'), index=False)
		
		pd.DataFrame({'sample_id': [sample_ids_valid[i] for i in range(len(sample_ids_valid)) if ~np.isnan(y_valid_gender[i]).any()],
					  'true': y_true_gender.tolist(), 
					  'pred_prob': y_pred_gender_str}).to_csv(os.path.join(args.output, 'Checkpoint_Oth', f'{args.model_type}_gender_fold{fold_idx + 1}_predictions.csv'), index=False)
		
		pd.DataFrame({'sample_id': [sample_ids_valid[i] for i in range(len(sample_ids_valid)) if ~np.isnan(y_valid_bmi[i]).any()],
					  'true': y_true_bmi.argmax(axis=1).tolist(), 
					  'pred_prob': y_pred_bmi_str}).to_csv(os.path.join(args.output, 'Checkpoint_Oth', f'{args.model_type}_bmi_fold{fold_idx + 1}_predictions.csv'), index=False)
		
		pd.DataFrame({'sample_id': [sample_ids_valid[i] for i in range(len(sample_ids_valid)) if ~np.isnan(y_valid_bodysite[i]).any()],
					  'true': y_true_bodysite.argmax(axis=1).tolist(), 
					  'pred_prob': y_pred_bodysite_str}).to_csv(os.path.join(args.output, 'Checkpoint_Oth', f'{args.model_type}_bodysite_fold{fold_idx + 1}_predictions.csv'), index=False)
		
		pd.DataFrame({'sample_id': [sample_ids_valid[i] for i in range(len(sample_ids_valid)) if ~np.isnan(y_valid_disease[i]).any()],
					  'true': y_true_disease.argmax(axis=1).tolist(), 
					  'pred_prob': y_pred_disease_str}).to_csv(os.path.join(args.output, 'Checkpoint_Oth', f'{args.model_type}_disease_fold{fold_idx + 1}_predictions.csv'), index=False)

		# Calculate evaluation metrics for each task
		age_acc = accuracy_score(y_true_age.argmax(axis=1), y_pred_age.argmax(axis=1))
		try:
			age_auc = roc_auc_score(y_true_age, y_pred_age, multi_class='ovr')
			age_f1 = f1_score(y_true_age.argmax(axis=1), y_pred_age.argmax(axis=1), average='macro')
			age_auprc = average_precision_score(y_true_age, y_pred_age, average='macro')
		except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
			print('When calculating the AUC score for age prediction, only one class is present in y_true: ')
			print(y_true_age.argmax(axis=1))
			print('ROC AUC score is not defined in that case.')
			age_auc = np.nan
			age_f1 = np.nan
			age_auprc = np.nan

		gender_acc = accuracy_score(y_true_gender, y_pred_gender.argmax(axis=1))
		try: 
			gender_auc = roc_auc_score(y_true_gender, y_pred_gender[:, 1])
			gender_f1 = f1_score(y_true_gender, y_pred_gender.argmax(axis=1))
			gender_auprc = average_precision_score(y_true_gender, y_pred_gender[:, 1], average='macro')
		except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
			print('When calculating the AUC score for gender prediction, only one class is present in y_true: ')
			print(y_true_gender.argmax(axis=1))
			print('ROC AUC score is not defined in that case.')
			gender_auc = np.nan
			gender_f1 = np.nan
			gender_auprc = np.nan

		bmi_acc = accuracy_score(y_true_bmi.argmax(axis=1), y_pred_bmi.argmax(axis=1))
		try: 
			bmi_auc = roc_auc_score(y_true_bmi, y_pred_bmi, multi_class='ovr')
			bmi_f1 = f1_score(y_true_bmi.argmax(axis=1), y_pred_bmi.argmax(axis=1), average='macro')
			bmi_auprc = average_precision_score(y_true_bmi, y_pred_bmi, average='macro')
		except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
			print('When calculating the AUC score for BMI prediction, only one class is present in y_true: ')
			print(y_true_bmi.argmax(axis=1))
			print('ROC AUC score is not defined in that case.')
			bmi_auc = np.nan
			bmi_f1 = np.nan
			bmi_auprc = np.nan

		bodysite_acc = accuracy_score(y_true_bodysite.argmax(axis=1), y_pred_bodysite.argmax(axis=1))
		try: 
			bodysite_auc = roc_auc_score(y_true_bodysite, y_pred_bodysite, multi_class='ovr')
			bodysite_f1 = f1_score(y_true_bodysite.argmax(axis=1), y_pred_bodysite.argmax(axis=1), average='macro')
			bodysite_auprc = average_precision_score(y_true_bodysite, y_pred_bodysite, average='macro')
		except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
			print('When calculating the AUC score for bodysite prediction, only one class is present in y_true: ')
			print(y_true_bodysite.argmax(axis=1))
			print('ROC AUC score is not defined in that case.')
			bodysite_auc = np.nan
			bodysite_f1 = np.nan
			bodysite_auprc = np.nan

		disease_acc = accuracy_score(y_true_disease.argmax(axis=1), y_pred_disease.argmax(axis=1))
		try: 
			disease_auc = roc_auc_score(y_true_disease, y_pred_disease, multi_class='ovr')
			disease_f1 = f1_score(y_true_disease.argmax(axis=1), y_pred_disease.argmax(axis=1), average='macro')
			disease_auprc = average_precision_score(y_true_disease, y_pred_disease, average='macro')
		except: # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
			print('When calculating the AUC score for disease prediction, only one class is present in y_true: ')
			print(y_true_disease.argmax(axis=1))
			print('ROC AUC score is not defined in that case.')
			disease_auc = np.nan
			disease_f1 = np.nan
			disease_auprc = np.nan

		# Update records
		records['Age ACC'].append(age_acc)
		records['Age AUC'].append(age_auc)
		records['Age F1'].append(age_f1)
		records['Age AUPRC'].append(age_auprc)
		records['Gender ACC'].append(gender_acc)
		records['Gender AUC'].append(gender_auc)
		records['Gender F1'].append(gender_f1)
		records['Gender AUPRC'].append(gender_auprc)
		records['BMI ACC'].append(bmi_acc)
		records['BMI AUC'].append(bmi_auc)
		records['BMI F1'].append(bmi_f1)
		records['BMI AUPRC'].append(bmi_auprc)
		records['Bodysite ACC'].append(bodysite_acc)
		records['Bodysite AUC'].append(bodysite_auc)
		records['Bodysite F1'].append(bodysite_f1)
		records['Bodysite AUPRC'].append(bodysite_auprc)
		records['Disease ACC'].append(disease_acc)
		records['Disease AUC'].append(disease_auc)
		records['Disease F1'].append(disease_f1)
		records['Disease AUPRC'].append(disease_auprc)
		records['Fold Index'].append(fold_idx)

	# Save records
	records_df = pd.DataFrame.from_dict(records)
	print(records_df)
	output_path = os.path.join(args.output, 'Checkpoint_Oth', f'{args.model_type}_records.csv')
	records_df.to_csv(output_path, index=False)
	print('Saved results to {}'.format(output_path))

if __name__ == '__main__':
	main()