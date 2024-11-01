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

	# Iterate over each fold
	for fold_idx, (train_indices, valid_indices) in enumerate(kfold.split(indices, dataset.diseases_idx)): 
		print(f'\nFold {fold_idx + 1}')

		# Initialize models and checkpoint paths 
		disease_model = None

		disease_checkpoint_path = os.path.join(args.output, 'Checkpoint_Oth0522', f'{args.model_type}_disease_model_fold{fold_idx + 1}.joblib')
		print('Check point paths: {}'.format(disease_checkpoint_path))

		# Prepare train and validation data for each task
		X_train = []
		y_train_age, y_train_gender, y_train_bmi, y_train_bodysite, y_train_disease = [], [], [], [], []

		for idx in train_indices:
			sample_id, spec, age, gender, bmi, bodysite, disease = dataset[idx]
			X_train.append(spec)
			y_train_disease.append(disease)

		X_train = np.stack(X_train, axis=0)
		y_train_disease = np.stack(y_train_disease, axis=0).argmax(axis=1)

		# Train models 
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
			y_valid_disease.append(disease)
			sample_ids_valid.append(sample_id)

		X_valid = np.stack(X_valid, axis=0)
		y_valid_disease = np.stack(y_valid_disease, axis=0)

		# Evaluate models for each task
		print('Evaluating models...')
		y_true_disease, y_pred_disease = eval_model(disease_model, X_valid[~np.isnan(y_valid_disease).any(axis=1)], y_valid_disease[~np.isnan(y_valid_disease).any(axis=1)])

		# Convert predicted probabilities to strings
		y_pred_disease_str = [','.join(map(str, prob)) for prob in y_pred_disease]

		# Save predictions to CSV
		pd.DataFrame({'sample_id': [sample_ids_valid[i] for i in range(len(sample_ids_valid)) if ~np.isnan(y_valid_disease[i]).any()],
					  'true': y_true_disease.argmax(axis=1).tolist(), 
					  'pred_prob': y_pred_disease_str}).to_csv(os.path.join(args.output, 'Checkpoint_Oth0522', f'{args.model_type}_disease_fold{fold_idx + 1}_predictions.csv'), index=False)

		# Calculate evaluation metrics for each task
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

if __name__ == '__main__':
	main()