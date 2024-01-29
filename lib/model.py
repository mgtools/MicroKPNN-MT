import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import pandas as pd



class MaskedLinear(nn.Module):
	def __init__(self, in_dim, out_dim, indices_mask):
		"""
		in_features: number of input features
		out_features: number of output features
		indices_mask: tensor used to create the mask
		"""
		super(MaskedLinear, self).__init__()
 
		def backward_hook(grad):
			# Clone due to not being allowed to modify in-place gradients
			out = grad.clone()
			out[indices_mask] = 0
			return out
 
		self.linear = nn.Linear(in_dim, out_dim)
		self.linear.weight.data[indices_mask] = 0 # zero out bad weights
		self.linear.weight.register_hook(backward_hook) # hook to zero out bad gradients
 
	def forward(self, x): 
		return self.linear(x)

class MicroKPNN_MTL(nn.Module): 
	def __init__(self, species, edge_list, in_dim, hidden_dim, bodysite_num, disease_num, mask): 
		super(MicroKPNN_MTL, self).__init__() 
		
		# generate the mask
		metadatas = ['BMI', 'gender', 'age', 'bodysite', 'phenotype']
		edge_df = pd.read_csv(edge_list)

		parent_nodes = list(set(edge_df['parent'].tolist()))
		parent_nodes = [node for node in parent_nodes if node not in metadatas] # remove metadata from parent nodes

		child_nodes = species 
		parent_dict = {k: i for i, k in enumerate(parent_nodes)}
		child_dict = {k: i for i, k in enumerate(child_nodes)}
		self.species_dict = child_dict # used outside the class
				
		for i, row in edge_df.iterrows():
			if row["parent"] not in metadatas and row['child'] != 'Unnamed: 0': 
				mask[child_dict[str(row['child'])]][parent_dict[row['parent']]] = 1
		mask = mask > 0 
		
		# establish the first customized linear
		assert in_dim == len(child_nodes), "Setting and edge list are not match, in_dim={}, but len(child_nodes)={}".format(in_dim, len(child_nodes))
		assert hidden_dim == len(parent_nodes), "Setting and edge list are not match, hidden_dim={}, but len(parent_nodes)={}".format(hidden_dim, len(parent_nodes))

		self.customized_linear = MaskedLinear(in_dim, hidden_dim, mask.permute(1, 0))
		
		self.decoder_age = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
										nn.ReLU(), 
										nn.Linear(hidden_dim, 6), 
										)
									
		self.decoder_gender = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
										nn.ReLU(), 
										nn.Linear(hidden_dim, 1), 
										nn.Sigmoid(), 
										)
		
		self.decoder_bmi = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
										nn.ReLU(),
										nn.Linear(hidden_dim, 4), 
										)
		
		self.decoder_bodysite = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
										nn.ReLU(), 
										nn.Linear(hidden_dim, bodysite_num), 
										)
		

		self.decoder_disease = nn.Sequential(nn.Linear(hidden_dim+bodysite_num+11, hidden_dim), 
										nn.ReLU(), 
										nn.Linear(hidden_dim, disease_num), 
										)

		for m in self.modules(): 
			if isinstance(m, (nn.Linear)): 
				nn.init.xavier_normal_(m.weight)
	
	def get_species_dict(self): 
		return self.species_dict

	def forward(self, x, real_age=None, real_gender=None, real_bmi=None, real_bodysite=None): 
		x = self.customized_linear(x)

		# meta-data prediction
		age = self.decoder_age(x)
		gender = self.decoder_gender(x)
		bmi = self.decoder_bmi(x)
		bodysite = self.decoder_bodysite(x)

		# replace the meta-data with their real values if real values is not None
		if real_age != None:
			mix_age = torch.where(torch.isnan(real_age), age, real_age)
		else:
			mix_age = age
		if real_gender != None:
			mix_gender = torch.where(torch.isnan(real_gender), gender, real_gender)
		else:
			mix_gender = gender
		if real_bmi != None:
			mix_bmi = torch.where(torch.isnan(real_bmi), bmi, real_bmi)
		else:
			mix_bmi = bmi
		if real_bodysite != None:
			mix_bodysite = torch.where(torch.isnan(real_bodysite), bodysite, real_bodysite)
		else:
			mix_bodysite = bodysite
	
		# disease prediction
		disease = self.decoder_disease(torch.cat((x, mix_age, mix_gender, mix_bmi, mix_bodysite), dim=1))

		# mix_age, mix_gender, and mix_bmi are only used to predict disease
		# we still need the predicted meta-data for training meta-data prediction
		return age, gender, bmi, bodysite, disease
