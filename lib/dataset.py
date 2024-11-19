import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import os
import pickle

from random import sample 



class MicroDataset(Dataset): 
	def __init__(self, data_path, metadata_path, species_dict, output): 
		"""
        Initialize MicroDataset with data and metadata paths, species dictionary, and output directory.
        If embedding data file exists, load it; otherwise, process the data and create the embedding.
        """

		embedding_file_path = output + "/embedding_data.pkl"
		if os.path.exists(embedding_file_path):
			self.load_embedding_data(output, embedding_file_path)

			# for debugging
			# self.sample_ids = sample(self.sample_ids, 2000)
			# self.species = sample(self.species, 2000)
			# self.ages = sample(self.ages, 2000)
			# self.genders = sample(self.genders, 2000)
			# self.bmis = sample(self.bmis, 2000)
			# self.bodysites = sample(self.bodysites, 2000)
			# self.diseases = sample(self.diseases, 2000)
			# self.diseases_idx = sample(self.diseases_idx, 2000)
		else:
			print(f"The file {embedding_file_path} does not exist.")
			self.process_data(data_path, metadata_path, species_dict, output)
		
			
	def load_embedding_data(self, output, embedding_file_path):
		"""
        Load embedding data from the specified file path and assign values to class attributes.
        """

		with open(embedding_file_path, 'rb') as file:
			loaded_embedding_data = pickle.load(file)

		self.sample_ids = loaded_embedding_data['sample_ids']
		self.species = loaded_embedding_data['species']
		self.ages = loaded_embedding_data['ages']
		self.genders = loaded_embedding_data['genders']
		self.bmis = loaded_embedding_data['bmis']
		self.bodysites = loaded_embedding_data['bodysites']
		self.diseases = loaded_embedding_data['diseases']
		self.diseases_idx = loaded_embedding_data['diseases_idx']

		self.age_dict = {'Infant': 0, 
				         'Children Adolescents': 1, 
						 'Young Adult': 2, 
						 'Middle Aged':3, 
						 'Senior':4, 
						 'Elderly':5, 
						 np.nan: np.nan}
		print('age encoding: {}'.format(self.age_dict))

		self.gender_dict = {'Female': 0, 'Male': 1, np.nan: np.nan} 
		print('gender encoding: {}'.format(self.gender_dict))
			
		self.bmi_dict = {'Underweight': 0, 
				         'Healthy Weight': 1, 
						 'Overweight': 2, 
						 'Obesity':3, 
						 np.nan: np.nan}
		print('bmi encoding: {}'.format(self.bmi_dict))

		bodysite_dict_file_path = os.path.join(output, "bodysitedict.csv")
		self.bodysite_dict = self.load_csv_dict(bodysite_dict_file_path)
		
		disease_dict_file_path = os.path.join(output, "diseasedict.csv")
		self.disease_dict = self.load_csv_dict(disease_dict_file_path)

		print('Embedding data loaded successfully.')
	

	def load_csv_dict(self, file_path):
		"""
        Load a CSV file as a dictionary and return the result.
        """
		result_dict = {}
		with open(file_path, 'r') as file:
			reader = csv.reader(file)
			for row in reader:
				key, val = row
				result_dict[key] = val
		return result_dict


	def process_data(self, data_path, metadata_path, species_dict, output):
		"""
        Process the data by merging species information, handling missing values, and creating encoding dictionaries.
        """
		df = pd.read_csv(data_path)
		meta_df = pd.read_csv(metadata_path)
		print('Load {} data from {}'.format(len(df), data_path))
		print('Load {} metadata from {}\n'.format(len(meta_df), metadata_path))

		# add lacked species columns to df
		new_columns = [k for k in species_dict.keys() if k not in list(df.columns)]
		df[new_columns] = 0. 
		
		# merge df & meta_df
		df = df.merge(meta_df, how='left', left_on='Unnamed: 0', right_on='run_id')
		
		self.age_dict = {'Infant': 0, 
				         'Children Adolescents': 1, 
						 'Young Adult': 2, 
						 'Middle Aged':3, 
						 'Senior':4, 
						 'Elderly':5, 
						 np.nan: np.nan}
		print('age encoding: {}'.format(self.age_dict))

		self.gender_dict = {'Female': 0, 'Male': 1, np.nan: np.nan} 
		print('gender encoding: {}'.format(self.gender_dict))
			
		self.bmi_dict = {'Underweight': 0, 
				         'Healthy Weight': 1, 
						 'Overweight': 2, 
						 'Obesity':3, 
						 np.nan: np.nan}
		print('bmi encoding: {}'.format(self.bmi_dict))

		self.bodysite_dict, bodysite_list = self.create_encoding_dict(output, meta_df, 'BodySite', 'bodysitedict.csv')
		print('bodysite encoding: {}\n'.format(self.bodysite_dict))

		self.disease_dict, disease_list = self.create_encoding_dict(output, meta_df, 'disease', 'diseasedict.csv')
		print('disease encoding: {}\n'.format(self.disease_dict))

		self.sample_ids, self.species, self.ages, self.genders, self.bmis, self.bodysites, self.diseases, self.diseases_idx = self.embedding_data(df, species_dict, bodysite_list, disease_list)
		
		embedding_data = {
    	'sample_ids': self.sample_ids,
    	'species': self.species,
    	'ages': self.ages,
    	'genders': self.genders,
    	'bmis': self.bmis,
		'bodysites': self.bodysites,
    	'diseases': self.diseases,
		'diseases_idx': self.diseases_idx
		}
		print('xxxxxx')
		print(len(embedding_data['sample_ids']))

		with open(output + '/embedding_data.pkl', 'wb') as file:
			pickle.dump(embedding_data, file)

		print('Embedding data saved to embedding_data.pkl')


	def create_encoding_dict(self, output, df, column, file_name):
		"""
        Create an encoding dictionary from a DataFrame column, write it to a CSV file, and return the dictionary.
        """
		column_list = list(set(df[column].dropna().tolist()))
		encoding_dict = {k: i for i, k in enumerate(column_list)}
		encoding_dict[np.nan] = np.nan
		
		with open(os.path.join(output, file_name), "w") as file:
			writer = csv.writer(file)
			for key, val in encoding_dict.items():
				writer.writerow([key, val])
		return encoding_dict, column_list
		
	
	def embedding_data(self, df, species_dict, bodysite_list, disease_list):
		"""
        Embed the data by converting categorical information into numerical representations.
        Return lists containing sample ids, species, ages, genders, bmis, bodysites, diseases, and diseases indices.
        """
		sample_ids, species, ages, genders, bmis, bodysites, diseases, diseases_idx = [], [], [], [], [], [], [], []
		
		print('Embedding the data...')
		for i, row in tqdm(df.iterrows(), total=df.shape[0]): 
			spec = row[species_dict.keys()].astype(float).tolist()
			
			sample_ids.append(row['run_id'])

			species.append(np.array(spec))
			
			age_idx = self.age_dict[row['host_age']]
			if np.isnan(age_idx):
				ages.append(np.array([age_idx]*6))
			else: 
				ages.append(self.one_hot(age_idx, num_classes=6))
			
			genders.append(np.array([self.gender_dict[row['sex']]]))
			
			bmi_idx = self.bmi_dict[row['BMI']]
			if np.isnan(bmi_idx):
				bmis.append(np.array([bmi_idx]*4))
			else: 
				bmis.append(self.one_hot(bmi_idx, num_classes=4))
			
			bodysite_idx = self.bodysite_dict[row['BodySite']]
			if np.isnan(bodysite_idx):
				bodysites.append(np.array([bodysite_idx]*(len(bodysite_list))))
			else: 
				bodysites.append(self.one_hot(bodysite_idx, num_classes=len(bodysite_list)))
				
			disease_idx = self.disease_dict[row['disease']]
			diseases_idx.append(disease_idx)
			if np.isnan(disease_idx):
				diseases.append(np.array([disease_idx]*(len(disease_list))))
			else: 
				diseases.append(self.one_hot(disease_idx, num_classes=len(disease_list)))

		return sample_ids, species, ages, genders, bmis, bodysites, diseases, diseases_idx
		

	def one_hot(self, idx, num_classes):
		one_hot_enc = np.zeros(num_classes)
		one_hot_enc[idx] = 1
		return one_hot_enc


	def get_age_dict(self, ):
		return self.age_dict


	def get_gender_dict(self, ):
		return self.gender_dict


	def get_bmi_dict(self, ):
		return self.bmi_dict


	def get_bodysite_dict(self, ):
		return self.bodysite_dict


	def get_disease_dict(self, ):
		return self.disease_dict


	def __getitem__(self, idx): 
		return self.sample_ids[idx], self.species[idx], self.ages[idx], self.genders[idx], self.bmis[idx], self.bodysites[idx], self.diseases[idx]


	def __len__(self): 
		return len(self.sample_ids)